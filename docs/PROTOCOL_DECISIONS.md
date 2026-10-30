# Reference binding and engineering decisions

Primary reference: the supplied `README.pdf` (51 physical pages). The original
`README.md` was a documentation placeholder, so it supplied no conflicting wire
requirements. These decisions apply to this implementation, not to an external
standard or an interoperability claim.

## MMQP framing and topics

1. The fixed header is one control octet followed by 1..4 **big-endian** remaining-
   length bytes. RLF encodes width minus one; this is not MQTT's variable integer.
   Encoders choose the shortest width and decoders reject nonminimal encodings,
   reserved bits, unknown types, invalid conditional fields, extra bytes and
   truncation. `max_packet_bytes` limits the complete encoded frame.
2. The main PUBREQ prose/table takes precedence over Table 14's `PRT=0` example:
   MESSAGE=1, REQUEST=2, RESPONSE=3. PRT zero and flag `0x10` are reserved/rejected.
   REQ/RES always carry a nonzero big-endian U16 correlation ID, independent of
   the optional QoS 1 packet ID. There is no separate correlation-present flag.
3. Topics are canonical UTF-8 and at most 255 encoded bytes. `+`, terminal `#`,
   shared groups, and the `$` namespace rules are checked before authorization,
   key derivation, routing and alias assignment. Leading/trailing separators and
   empty levels are removed. Case folding and Unicode NFC equivalence are not
   invented; the same canonical form is used for routing and encryption.
4. Aliases are session-local and directional. Publisher aliases use 1..65535;
   client-declared incoming aliases use 1..32767; broker wildcard assignments use
   32768..65535. An existing high incoming alias can be used in a subsequent
   concrete subscription/unsubscription; a client cannot invent a new high one.
   No alias may be rebound. Topic strings remain present until confirmation.
5. TIE is an exact multiplier 1..63 with seconds/minutes/hours/days tiers. A
   configured duration that cannot be represented exactly is rejected, not rounded.
   Retention implements the documented none/latest/FIFO queue semantics and expiry.

## Reliability and exchanges

QoS 0 has no MMQP acknowledgement. QoS 1 acknowledges admission at the current hop,
after required durable state is committed. It does not imply that every downstream
subscriber application has completed. Retries retain their PID and set DUP. The
receiver verifies duplicate content using a canonical digest. PIDs are independent
in each direction and wrap over the nonzero 16-bit space, while internal sequences
are 64-bit. Ordered transport and QUIC reordering use distinct receive-window
handling; both retain a 256-publication horizon.

Profile-request IDs have a separate nonwrapping 16-bit sequence. The client
serializes profile transactions; after 65535 updates it requires a fresh session
rather than reinterpreting old profile receipts. The reference separates this
sequence from publication PIDs but does not specify its wrap binding. This is an
explicit conservative lifetime limit, not a limit on publication throughput.

Request IDs are scoped to the requester session and reusable after completion.
The broker allocates separate responder-facing IDs and persists their mapping to
requester SID, original ID, canonical topic, admitted responder and deadline. A
responder cannot answer another session's request or use a different topic. One
response per admitted responder consumes that route; an application error counts
as that responder's outcome rather than suppressing other admitted replies.

The maximum response count bounds admission; zero means all eligible responders.
The wire format has no final-admitted-count/completion frame. Therefore the client
can release a finite exchange after that many responses, but waits until
`response_timeout_ms` when fewer responders were admitted or count was zero.
The broker can release its own state earlier once its admitted set is exhausted.
The local deadline generates a correlated timeout outcome and releases the ID.
Choose it longer than the broker response deadline and expected failover interval.
A response to an inactive/unknown ID is rejected. The allocator moves forward
through unused IDs, but an application that explicitly reuses an ID while an old
peer can still respond creates ambiguity: there is no wire generation counter.
Avoid such reuse until the previous exchange's response/retry lifetime has ended.

No-responder and timeout outcomes are empty correlated RESP packets. PUBACK still
reports successful request admission when the broker has generated that outcome;
mixing an unsuccessful PUBACK with the RESP would create an avoidable cross-stream
race. Empty unencrypted broker errors are permitted when the application requires
encrypted payloads. Nonempty application data must satisfy its encryption policy.

Reason assignments not fixed by the manuscript are an explicit binding:

| Code | Meaning | Code | Meaning |
| --- | --- | --- | --- |
| `00` | Success | `80` | Unspecified failure |
| `01` | No subscriber (accepted) | `81` | Authentication failed |
| `02` | Alias mapped | `82` | Banned |
| `83` | Policy violation | `84` | Malformed packet |
| `85` | Session expired | `86` | Unauthorized |
| `87` | Keep-alive timeout | `88` | Unsupported version |
| `89` | Invalid topic | `8A` | Invalid alias |
| `8B` | Invalid packet ID | `8C` | Internal error |
| `8D` | No responder | `8E` | Busy / exhausted budget |
| `8F` | Timeout | `90` | Invalid correlation |

Reason strings appear only under negotiated IRS and a nonzero reason. Errors are
codes at least `0x80`; informational success codes are preserved by futures/events.

## Authentication and payload protection

TLS uses 1.3 as the minimum supported binding, a stricter choice than the PDF's
TLS 1.2+ illustration. Clients verify authority and hostname; clustered peers also
verify manifest certificate pins and require mutual authentication. Loopback TCP
is the development default. Remote plaintext and insecure application
authentication require explicit configuration rather than accidental exposure.

The challenge exchange uses secure random tokens, HMAC-SHA256 and replay tracking.
Tenant selection may prefix a token with the first eight bytes of
`SHA256("MMQP/AUTH/" || identity)`; this selector is not itself authentication.
Tenant roots and shard restrictions are enforced against authenticated identity.
Authentication keys, fabric MAC keys and payload keys have separate provisioning.

The current **Ascon-AEAD128** specification takes precedence over legacy ASCON-128
labels in the illustrations. The implementation uses the pinned upstream
[Ascon C reference](https://github.com/ascon/ascon-c) for
[NIST SP 800-232](https://csrc.nist.gov/pubs/sp/800/232/final), with a 16-byte key,
16-byte nonce and 16-byte tag. The vendor README records the commit and license.
All 1089 supplied upstream known-answer vectors are exercised by `Security`.

Hierarchical HKDF derivation is domain-separated by canonical levels and epoch.
A grant is a root secret or an epoch-scoped subtree secret; it cannot derive a
sibling/ancestor key. Associated data is canonical topic bytes followed by the
big-endian U32 epoch. Topic authorization and correlation checks remain broker
responsibilities: outer routing metadata is not included in payload AEAD.

Nonces use a random 64-bit sender prefix and a 64-bit counter. With
`payload_state_directory`, counter ranges are durably reserved before encryption,
and replay windows survive restart. Without it, safety across process restarts
depends on a fresh random prefix and replay history is process-local. Each sender
must have exclusive state; do not clone/roll back a state directory or key/nonce
image. Epoch policy forbids rollback, and capacity exhaustion fails closed rather
than evicting still-needed replay protection. External KMS/key distribution is a
provisioning boundary, not an invented network service.

## QUIC

Allocation values follow the current PDF: `01` single, `02` semantic, `03` semantic
plus bulk. `00` is reserved for an application-defined policy and is rejected by
this binding. Control is client-initiated bidirectional stream 0. Unidirectional
streams start with class marker 1 (QoS 0), 2 (QoS 1), or 3 (one bulk frame followed
by FIN). Bulk credits and pre-negotiation buffering are bounded. Large or
temporarily ineligible publications fall back to the negotiated reliable stream.

Datagrams carry only complete eligible QoS 0 publications and fit the negotiated
payload threshold and the native datagram limit. They add no MMQP reliability.
Session policy is immutable across resume, and 0-RTT application data is rejected.
Changing transport/policy requires a fresh session.

## Fabric and persistence

The fabric uses static authenticated membership, deterministic SHA-256-derived
placement and forwarding trees, epoch-bounded MAC envelopes, majority leases and
incarnation/fence checks. MHF/1 is the forwarding envelope; MHF/2 is the private
control RPC binding. Live profiles use versioned replace-per-session/per-shard
snapshots with empty tombstones; the reusable observed-remove profile primitive
is also implemented/tested. A replacement snapshot prevents stale per-entry
assertions from reappearing after unsubscribe/recovery.

Lease holders use a conservative `L/3` interval within voters' `L` grants. A
restarted voter waits `L` before voting; clocks advance during suspend. Recovery
adopts a majority-certified snapshot and commits a new fence before serving.
Notifications are hints; bounded inbox polling recovers missed notifications.
Durable recipient inboxes, consumed tombstones and request routes permit replay
without treating an old owner as authoritative. Trust, quorum and storage
assumptions are detailed in [CLUSTERING.md](CLUSTERING.md).

Current snapshots are **ESS6**, **EBR6**, and manifest schema **6**, reflecting
reusable request/responder ID allocation. The outer broker journal remains EBJ1.
Older state/schema and conflicting membership/epoch directories are rejected.
No automatic on-disk or mixed-peer wire migration is supplied. Retain matched
peers and explicitly provision/migrate a deployment when these bindings change.
