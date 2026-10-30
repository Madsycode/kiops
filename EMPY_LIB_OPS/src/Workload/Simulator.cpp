#include "RuntimeDetail.h"
#include <eOps/Common/Text.h>
#include <eOps/Common/TopicAlgebra.h>
#include <eCore/Common/Stdout.h>
#include <stdexcept>
#include <numbers>
#include <cmath>

namespace eOps::Workload
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        // ground truth y = a·x1 + b·x2 + c over the first two observables; the drifted
        // coefficients model a concept drift that keeps y inside the home/set/temp range
        constexpr double BASE_A = 0.5, BASE_B = 0.05, BASE_C = 8;
        constexpr double DRIFT_A = 0.2, DRIFT_B = 0.1, DRIFT_C = 12;
        // angular step per tick of the first signal; later signals are slower by the golden ratio so
        // features never become collinear
        constexpr double BASE_FREQUENCY = 0.2;

        uint64_t SplitMix(uint64_t value)
        {
            value += 0x9E3779B97F4A7C15ull;
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
            return value ^ (value >> 31);
        }

        bool Contains(std::string_view text, std::string_view part)
        {
            return text.find(part) != std::string_view::npos;
        }
    } // namespace

    struct Simulator::Impl
    {
        // one synthetic sensor: base + amplitude·sin(frequency·tick + phase)
        struct Signal
        {
            std::string m_Topic;
            std::string m_Unit;
            uint8_t m_Qos = 1;
            double m_Base = 10;
            double m_Amplitude = 5;
            double m_Frequency = BASE_FREQUENCY;
            double m_Phase = 0;
        };

        SimulatorOptions m_Options;
        std::vector<Signal> m_Signals;
        std::string m_Label;
        std::string m_Locality;
        uint64_t m_Tick = 0;
        Clock::time_point m_Started;
        bool m_Drifted = false;
        eMQ::Service *m_Service = nullptr;
        Detail::Ticker m_Period;
        Detail::Ticker m_Report;
        Detail::Outbox m_Data;    // observable + label publications
        Detail::Outbox m_Control; // locality reports

        explicit Impl(SimulatorOptions options) : m_Options(std::move(options))
        {
            const auto &service = m_Options.m_Service;
            if (!m_Options.m_PeriodMs)
            {
                throw std::invalid_argument("Simulator period must be at least 1 ms");
            }
            m_Locality = m_Options.m_Topics.Locality(m_Options.m_Region); // validates the region level
            for (const auto &observable : service.m_Observables)
            {
                if (!ValidTopic(observable.m_Name, false) || IsFilter(observable.m_Name))
                {
                    continue; // wildcard observables have no single topic to publish on
                }
                Signal signal;
                signal.m_Topic = observable.m_Name;
                signal.m_Unit = observable.m_Unit;
                signal.m_Qos = observable.m_Qos > 1 ? 1 : observable.m_Qos;
                std::string unit = Lower(observable.m_Unit);
                std::string leaf = Lower(observable.m_Name.substr(observable.m_Name.rfind('/') + 1));
                if (unit == "celsius" || unit == "celcius" || Contains(leaf, "temp"))
                {
                    signal.m_Base = 20;
                    signal.m_Amplitude = 3;
                }
                else if (unit == "percent" || Contains(leaf, "hum"))
                {
                    signal.m_Base = 50;
                    signal.m_Amplitude = 10;
                }
                size_t index = m_Signals.size();
                signal.m_Frequency = BASE_FREQUENCY / std::pow(std::numbers::phi, static_cast<double>(index));
                uint64_t random = SplitMix(m_Options.m_Seed + 0x632BE59BD9B4E019ull * (index + 1));
                signal.m_Phase = static_cast<double>(random >> 11) * 0x1.0p-53 * 2 * std::numbers::pi;
                m_Signals.push_back(std::move(signal));
            }
            if (m_Signals.empty())
            {
                throw std::invalid_argument("Simulator needs at least one concrete observable");
            }
            if (service.m_Training && !service.m_Training->m_Label.empty())
            {
                m_Label = service.m_Training->m_Label;
                if (!ValidTopic(m_Label, false) || IsFilter(m_Label))
                {
                    throw std::invalid_argument("training.label must be a concrete topic: " + m_Label);
                }
            }
        }

        static eCore::SharedBytes Value(double value, std::string_view unit, uint64_t time)
        {
            Json json = Json::Object{{"value", value}, {"time", time}};
            if (!unit.empty())
            {
                json.Set("unit", unit);
            }
            return ToBytes(json);
        }

        void Publish(Detail::Outbox &outbox, const std::string &topic, eCore::SharedBytes payload, uint8_t qos)
        {
            auto *bus = Detail::FindBus(m_Service);
            if (!bus)
            {
                return;
            }
            outbox.Send(topic, [&] { return bus->PublishRaw(topic, std::move(payload), qos); });
        }

        void Tick()
        {
            m_Data.Harvest();
            if (!m_Drifted && m_Options.m_DriftAfterMs &&
                Clock::now() - m_Started >= std::chrono::milliseconds(m_Options.m_DriftAfterMs))
            {
                m_Drifted = true;
                EMPY_INFO("Simulator {}: concept drift induced after {} ticks", m_Options.m_Service.m_Metadata.m_Name,
                          m_Tick);
            }
            const double step = static_cast<double>(m_Tick++);
            const uint64_t now = NowMillis();
            std::vector<double> values;
            values.reserve(m_Signals.size());
            for (const auto &signal : m_Signals)
            {
                double value =
                    signal.m_Base + signal.m_Amplitude * std::sin(signal.m_Frequency * step + signal.m_Phase);
                values.push_back(value);
                this->Publish(m_Data, signal.m_Topic, Value(value, signal.m_Unit, now), signal.m_Qos);
            }
            if (!m_Label.empty())
            {
                double x1 = values[0];
                double x2 = values.size() > 1 ? values[1] : 0;
                double label = m_Drifted ? DRIFT_A * x1 + DRIFT_B * x2 + DRIFT_C : BASE_A * x1 + BASE_B * x2 + BASE_C;
                this->Publish(m_Data, m_Label, Value(label, {}, now), 1);
            }
        }

        void Report()
        {
            m_Control.Harvest();
            auto *bus = Detail::FindBus(m_Service);
            if (!bus)
            {
                return;
            }
            Json topics = Json::MakeObject();
            for (const auto &signal : m_Signals)
            {
                topics.Set(signal.m_Topic, m_Options.m_LocalityRate);
            }
            auto message =
                Fabric::Envelope::Make("locality", Json::Object{{"region", m_Options.m_Region},
                                                                {"service", m_Options.m_Service.m_Metadata.m_Name},
                                                                {"topics", std::move(topics)}});
            m_Control.Send(m_Locality, [&] { return bus->Publish(m_Locality, message, 1); });
        }

        void Attach(eMQ::Service *service)
        {
            m_Service = service;
            auto &bus = Fabric::BusOf(*service);
            // publication profiles first, so the first tick never blocks on a registration
            for (const auto &signal : m_Signals)
            {
                Detail::Require(bus.Expose(signal.m_Topic), "Publication profile " + signal.m_Topic);
            }
            if (!m_Label.empty())
            {
                Detail::Require(bus.Expose(m_Label), "Publication profile " + m_Label);
            }
            Detail::Require(bus.Expose(m_Locality), "Locality profile");
            m_Started = Clock::now();
            m_Period.Start(*service, m_Options.m_PeriodMs, [this] { this->Tick(); });
            m_Report.Start(*service, m_Options.m_LocalityMs, [this] { this->Report(); });
            EMPY_INFO("Simulator {}: {} signals every {} ms", m_Options.m_Service.m_Metadata.m_Name, m_Signals.size(),
                      m_Options.m_PeriodMs);
            if (m_Options.m_LocalityMs)
            {
                this->Report();
            }
        }

        void Detach()
        {
            m_Period.Stop();
            m_Report.Stop();
            m_Service = nullptr;
        }
    };

    Simulator::Simulator(SimulatorOptions options) : m_Impl(std::make_unique<Impl>(std::move(options)))
    {
    }

    Simulator::~Simulator() = default;

    size_t Simulator::Published() const
    {
        return static_cast<size_t>(m_Impl->m_Data.Accepted());
    }

    bool Simulator::Drifted() const
    {
        return m_Impl->m_Drifted;
    }

    void Simulator::OnAttach(eMQ::Service *service, eMQ::Context *)
    {
        m_Impl->Attach(service);
    }

    void Simulator::OnUpdate(eMQ::Service *, eMQ::Context *)
    {
        m_Impl->m_Data.Harvest();
        m_Impl->m_Control.Harvest();
    }

    void Simulator::OnDetach(eMQ::Service *, eMQ::Context *)
    {
        m_Impl->Detach();
    }
} // namespace eOps::Workload
