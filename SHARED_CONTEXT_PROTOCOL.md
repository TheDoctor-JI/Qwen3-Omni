# Context-scoped audio views (protocol 1)

The new client uses the existing Socket.IO `generate`/`stop` and streamed response
events. Legacy inline/base64 and socket-local cache requests remain supported.
The new route shares prepared audio across independent generation connections.
It requires the sibling `mm_llm_interface` package's lightweight protocol helpers.

## Owner control connection

The context owner (eventually the context manager; a standalone caller in tests)
opens its own socket and calls:

```text
register_context {process_id: <manager-process-id>, interaction_id: <interaction-id>}
  -> {ok: true, process_id, interaction_id}
retire_context {process_id, interaction_id}
  -> {ok: true}
```

Registration is idempotent for the owning connection. Another connection cannot
claim or retire that namespace. Generation clients only attach by including the
identity in their requests. Their disconnect does not clear shared entries.
Owner retirement/disconnect retires the cache and aborts attached generations;
it does not affect another namespace. Retired identities cannot be reused.

## Generation payload

```json
{
  "protocol_version": 1,
  "context": {"process_id": "manager-id", "interaction_id": "interaction-id"},
  "request_id": "independent-generation-id",
  "messages": [{"role": "user", "content": [{
    "type": "audio",
    "item_id": "speech-id",
    "view_key": "<deterministic-sha256>",
    "media": {
      "source_id": "speech-id",
      "start_frame": 160,
      "end_frame": 1600,
      "sample_rate": 16000,
      "channels": 1,
      "encoding": "s16le",
      "blocks": [{"path": "/dev/shm/.../block", "start": 0,
                  "length": 3200, "capacity": 1048576}]
    }
  }]}],
  "params": {"thinking_mode": false, "temperature": 0.6, "max_tokens": 8192}
}
```

Frame ranges are half-open and relative to the logical media assembled from
published block lengths. They are fixed by the snapshot producer. The server does
not infer playback/cancellation limits. It validates the view hash with
`mm_llm_interface.protocol.view_key`, maps the exact range on a cache miss,
converts PCM16 to a float waveform, and runs normal `process_mm_info` preparation
in memory. No temporary WAV or base64 upload is needed for this route.

The logical cache key is namespace -> item ID -> view key. The view key includes
source identity, exact exposed frame boundaries, format and preprocessing version.
Appending future media does not invalidate a previously exposed immutable prefix.
Two views of equal duration but different ranges cannot alias. Concurrent requests
share synchronized preparation; retirement fences late worker-thread insertions.

`items_cached` acknowledges `{item_id, meta_key}` entries, where `meta_key` is the
view hash for this route. Every request includes descriptors even if a view is
expected to be cached, allowing the server to resolve a miss directly. Cache scope
does not change generation ownership: a new generation replaces the previous one
only on its own socket, and `stop` carries the request ID.

The application cache stores prepared media, not transformer KV or GPU encoder
tensors. vLLM's own content-hash encoder cache remains responsible for embedding
reuse and eviction. The client/context owner retains backing files or snapshot
leases while requests can read them. Final context end may abort those requests.

Qwen and the backing-media owner must run on the same host with access to the
same mappings. Speech inputs for this first implementation are mono 16 kHz PCM16;
remote media transfer and additional modalities on this route are outside scope.
