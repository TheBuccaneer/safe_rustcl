// Dummy Buffer-Typ ohne OpenCL-Abhängigkeit
struct DummyBuffer(u64);

// Typ-State-Dummy-Buffer
struct DummyGpuBuffer<S> {
    buf: DummyBuffer,
    len: usize,
    _state: std::marker::PhantomData<S>,
}

// Typ-State-Marker
mod sealed { pub trait Sealed {} }
trait State: sealed::Sealed {}

struct Queued;
struct InFlight;
struct Ready;
impl sealed::Sealed for Queued {}
impl sealed::Sealed for InFlight {}
impl sealed::Sealed for Ready {}
impl State for Queued {}
impl State for InFlight {}
impl State for Ready {}

#[test]
fn typestate_transitions_dummy_only() {
    // Dummy erzeugen ohne undefiniertes Verhalten
    let dummy = DummyBuffer(12345);

    let queued = DummyGpuBuffer::<Queued> {
        buf: dummy,
        len: 42,
        _state: std::marker::PhantomData,
    };

    let inflight: DummyGpuBuffer<InFlight> = DummyGpuBuffer {
        buf: queued.buf,
        len: queued.len,
        _state: std::marker::PhantomData,
    };

    struct DummyGuard;
    impl Drop for DummyGuard {
        fn drop(&mut self) {}
    }

    let guard = DummyGuard;

    let _ready: DummyGpuBuffer<Ready> = DummyGpuBuffer {
        buf: inflight.buf,
        len: inflight.len,
        _state: std::marker::PhantomData,
    };

    drop(guard); // DummyGuard Drop korrekt durchlaufen
}
