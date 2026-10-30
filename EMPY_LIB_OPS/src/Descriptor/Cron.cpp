#include <eOps/Descriptor/Cron.h>
#include <eOps/Common/Text.h>
#include <array>
#include <limits>
#include <charconv>
#include <stdexcept>

namespace eOps::Descriptor
{
    namespace
    {
        using namespace std::chrono;

        constexpr std::array<std::string_view, 12> MONTHS = {"jan", "feb", "mar", "apr", "may", "jun",
                                                             "jul", "aug", "sep", "oct", "nov", "dec"};
        constexpr std::array<std::string_view, 7> WEEKDAYS = {"sun", "mon", "tue", "wed", "thu", "fri", "sat"};
        // the longest month a day-of-month can fall in (February admits the 29th in leap years)
        constexpr std::array<uint32_t, 13> MONTH_DAYS = {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

        struct Field
        {
            const char *m_Name;
            uint32_t m_Minimum;
            uint32_t m_Maximum;
            const std::string_view *m_Names = nullptr; // names[i] = value m_Minimum + i
            size_t m_NameCount = 0;
        };

        [[noreturn]] void Invalid(std::string_view expression, const std::string &reason)
        {
            throw std::invalid_argument("invalid cron expression '" + std::string(expression) + "': " + reason);
        }

        uint32_t Value(std::string_view text, const Field &field, std::string_view expression)
        {
            for (size_t i = 0; i < field.m_NameCount; ++i)
            {
                if (text == field.m_Names[i])
                {
                    return field.m_Minimum + static_cast<uint32_t>(i);
                }
            }
            uint32_t value = 0;
            auto [end, code] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (text.empty() || code != std::errc() || end != text.data() + text.size())
            {
                Invalid(expression, std::string(field.m_Name) + " value '" + std::string(text) + "'");
            }
            if (value < field.m_Minimum || value > field.m_Maximum)
            {
                Invalid(expression, std::string(field.m_Name) + " value " + std::to_string(value) + " outside [" +
                                        std::to_string(field.m_Minimum) + ", " + std::to_string(field.m_Maximum) + "]");
            }
            return value;
        }

        // bit set of one field: comma list of '*', 'a', 'a-b', each optionally '/step'
        uint64_t ParseField(std::string_view text, const Field &field, std::string_view expression)
        {
            uint64_t bits = 0;
            for (const std::string &item : Split(text, ','))
            {
                std::string_view range = item;
                uint32_t step = 1;
                size_t slash = range.find('/');
                if (slash != std::string_view::npos)
                {
                    std::string_view stepText = range.substr(slash + 1);
                    auto [end, code] = std::from_chars(stepText.data(), stepText.data() + stepText.size(), step);
                    if (stepText.empty() || code != std::errc() || end != stepText.data() + stepText.size() ||
                        step == 0 || step > field.m_Maximum + 1)
                    {
                        Invalid(expression, std::string(field.m_Name) + " step '" + std::string(stepText) + "'");
                    }
                    range = range.substr(0, slash);
                }
                uint32_t first = field.m_Minimum;
                uint32_t last = field.m_Maximum;
                if (range != "*")
                {
                    size_t dash = range.find('-');
                    first = Value(range.substr(0, dash), field, expression);
                    if (dash != std::string_view::npos)
                    {
                        last = Value(range.substr(dash + 1), field, expression);
                    }
                    else if (slash == std::string_view::npos)
                    {
                        last = first; // a single value; "a/step" runs to the field maximum (Vixie)
                    }
                    if (first > last)
                    {
                        Invalid(expression, std::string(field.m_Name) + " range '" + std::string(range) + "'");
                    }
                }
                for (uint32_t value = first; value <= last; value += step)
                {
                    bits |= uint64_t{1} << value;
                }
            }
            return bits;
        }

        // the next minute boundary strictly after `after`
        sys_time<minutes> FirstCandidate(system_clock::time_point after)
        {
            return floor<minutes>(after) + minutes(1);
        }
    } // namespace

    CronSchedule CronSchedule::Parse(std::string_view expression)
    {
        std::string text = Lower(Trim(expression));
        CronSchedule schedule;
        schedule.m_Expression = Trim(expression);
        if (text.empty())
        {
            Invalid(expression, "empty");
        }
        if (text.starts_with("@every"))
        {
            std::string period = Trim(std::string_view(text).substr(6));
            if (period.size() < 2 || text.size() < 7 || (text[6] != ' ' && text[6] != '\t'))
            {
                Invalid(expression, "expected '@every <N>(s|m|h)'");
            }
            char unit = period.back();
            uint64_t factor = unit == 's' ? 1000 : unit == 'm' ? 60000 : unit == 'h' ? 3600000 : 0;
            uint64_t count = 0;
            auto [end, code] = std::from_chars(period.data(), period.data() + period.size() - 1, count);
            if (factor == 0 || code != std::errc() || end != period.data() + period.size() - 1 || count == 0 ||
                count > std::numeric_limits<uint64_t>::max() / factor / 1000)
            {
                Invalid(expression, "expected '@every <N>(s|m|h)' with N >= 1");
            }
            schedule.m_EveryMillis = count * factor;
            return schedule;
        }
        if (text.front() == '@')
        {
            static const std::array<std::pair<std::string_view, std::string_view>, 7> macros = {{
                {"@yearly", "0 0 1 1 *"},
                {"@annually", "0 0 1 1 *"},
                {"@monthly", "0 0 1 * *"},
                {"@weekly", "0 0 * * 0"},
                {"@daily", "0 0 * * *"},
                {"@midnight", "0 0 * * *"},
                {"@hourly", "0 * * * *"},
            }};
            bool found = false;
            for (const auto &[name, expansion] : macros)
            {
                if (text == name)
                {
                    text = expansion;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                Invalid(expression, "unknown macro");
            }
        }

        std::vector<std::string> fields;
        for (const std::string &part : Split(text, ' '))
        {
            for (const std::string &piece : Split(part, '\t'))
            {
                if (!piece.empty())
                {
                    fields.push_back(piece);
                }
            }
        }
        if (fields.size() != 5)
        {
            Invalid(expression, "expected 5 fields (minute hour day-of-month month day-of-week)");
        }
        const Field minute{"minute", 0, 59};
        const Field hour{"hour", 0, 23};
        const Field day{"day-of-month", 1, 31};
        const Field month{"month", 1, 12, MONTHS.data(), MONTHS.size()};
        const Field weekday{"day-of-week", 0, 7, WEEKDAYS.data(), WEEKDAYS.size()};
        schedule.m_Minutes = ParseField(fields[0], minute, expression);
        schedule.m_Hours = static_cast<uint32_t>(ParseField(fields[1], hour, expression));
        schedule.m_Days = static_cast<uint32_t>(ParseField(fields[2], day, expression));
        schedule.m_Months = static_cast<uint16_t>(ParseField(fields[3], month, expression));
        uint64_t weekdays = ParseField(fields[4], weekday, expression);
        if (weekdays & (uint64_t{1} << 7))
        {
            weekdays = (weekdays | 1) & 0x7f; // 7 is Sunday as well
        }
        schedule.m_Weekdays = static_cast<uint8_t>(weekdays);
        // Vixie cron: a field starting with '*' counts as unrestricted (also "*/2")
        schedule.m_DayRestricted = fields[2].front() != '*';
        schedule.m_WeekdayRestricted = fields[4].front() != '*';

        // reject schedules that can never fire (e.g. "0 0 30 2 *") so Next always terminates
        if (schedule.m_DayRestricted && !schedule.m_WeekdayRestricted)
        {
            bool possible = false;
            for (uint32_t m = 1; m <= 12 && !possible; ++m)
            {
                if (schedule.m_Months & (1u << m))
                {
                    for (uint32_t d = 1; d <= MONTH_DAYS[m] && !possible; ++d)
                    {
                        possible = (schedule.m_Days >> d) & 1;
                    }
                }
            }
            if (!possible)
            {
                Invalid(expression, "no month contains the requested day");
            }
        }
        return schedule;
    }

    system_clock::time_point CronSchedule::Next(system_clock::time_point after) const
    {
        if (m_EveryMillis)
        {
            int64_t now = duration_cast<milliseconds>(after.time_since_epoch()).count();
            if (milliseconds(now) > after.time_since_epoch())
            {
                --now; // floor for times before the epoch
            }
            int64_t period = static_cast<int64_t>(m_EveryMillis);
            int64_t slot = now >= 0 ? now / period : -((-now + period - 1) / period);
            return system_clock::time_point(duration_cast<system_clock::duration>(milliseconds((slot + 1) * period)));
        }

        sys_time<minutes> candidate = FirstCandidate(after);
        // every restricted combination fires within a leap cycle; the bound only guards against defects
        for (uint32_t iteration = 0; iteration < 1000000; ++iteration)
        {
            sys_days date = floor<days>(candidate);
            year_month_day ymd(date);
            uint32_t month = static_cast<uint32_t>(ymd.month());
            if (!((m_Months >> month) & 1))
            {
                year_month next = year_month(ymd.year(), ymd.month()) + months(1);
                candidate = sys_days(next / 1);
                continue;
            }
            uint32_t day = static_cast<uint32_t>(ymd.day());
            uint32_t dow = std::chrono::weekday(date).c_encoding();
            bool dayMatch = (m_Days >> day) & 1;
            bool weekdayMatch = (m_Weekdays >> dow) & 1;
            // Vixie cron: OR when both day fields are restricted, else AND (a '*' field has every bit set)
            bool match =
                m_DayRestricted && m_WeekdayRestricted ? (dayMatch || weekdayMatch) : (dayMatch && weekdayMatch);
            if (!match)
            {
                candidate = date + days(1);
                continue;
            }
            hh_mm_ss<minutes> time(candidate - date);
            uint32_t hour = static_cast<uint32_t>(time.hours().count());
            if (!((m_Hours >> hour) & 1))
            {
                candidate = floor<hours>(candidate) + hours(1);
                continue;
            }
            uint32_t minute = static_cast<uint32_t>(time.minutes().count());
            uint32_t next = minute;
            while (next < 60 && !((m_Minutes >> next) & 1))
            {
                ++next;
            }
            if (next == 60)
            {
                candidate = floor<hours>(candidate) + hours(1);
                continue;
            }
            return time_point_cast<system_clock::duration>(candidate + minutes(next - minute));
        }
        throw std::runtime_error("cron expression '" + m_Expression + "' has no firing time");
    }

    const std::string &CronSchedule::Expression() const noexcept
    {
        return m_Expression;
    }
} // namespace eOps::Descriptor
