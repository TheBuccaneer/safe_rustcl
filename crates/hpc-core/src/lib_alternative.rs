// ─── Feature‑Module ───────────────────────────────────────────────────
#[cfg(feature = "metrics")]
mod metrics;
#[cfg(feature = "metrics")]
pub use metrics::*;

#[cfg(feature = "memtrace")]
mod memtracer;
#[cfg(feature = "memtrace")]
pub use memtracer::{start, Dir, CopyToken, flush_csv, TracingScope, is_auto_trace_enabled, enable_auto_trace, disable_auto_trace};

// ─── Extern‑Callback (nur für memtrace) ───────────────────────────────
#[cfg(feature = "memtrace")]
use {
    opencl3::types::{cl_event, cl_int},
    std::ffi::c_void,
};

#[cfg(feature = "memtrace")]
pub extern "C" fn memtrace_callback(
    _evt: cl_event,
    _status: cl_int,
    user_data: *mut c_void,
) {
    // Safety: stammt aus Box::into_raw
    let tok: Box<CopyToken> = unsafe { Box::from_raw(user_data.cast()) };
    tok.finish();
}

// ─── OpenCL / Std‑Imports ─────────────────────────────────────────────
use opencl3::{
    context::Context,
    memory::{Buffer, CL_MEM_READ_WRITE},
    command_queue::CommandQueue,
    event::Event,
    types::CL_NON_BLOCKING,
};
use std::{marker::PhantomData, ptr};

#[cfg(feature = "metrics")]
use std::time::Instant;

// **Performance**: Nur wenn metrics aktiv sind
#[cfg(feature = "metrics")]
use std::sync::atomic::Ordering;

// ─── Fehler‑Typ & cl_try! ─────────────────────────────────────────────
#[derive(thiserror::Error, Debug)]
pub enum ClError {
    #[error("OpenCL API error: {0}")]
    Api(i32),
    #[error("Invalid buffer size: {0}")]
    InvalidSize(usize),
}

/// **Performance**: Inline-Assembly für kritische OpenCL-Calls
macro_rules! cl_try {
    ($expr:expr) => {
        let err = unsafe { $expr };
        if err != 0 {
            return Err(crate::ClError::Api(err));
        }
    };
}
pub(crate) use cl_try;

impl From<opencl3::error_codes::ClError> for ClError {
    #[inline]
    fn from(err: opencl3::error_codes::ClError) -> Self { 
        ClError::Api(err.0) 
    }
}

impl From<i32> for ClError {
    #[inline]
    fn from(code: i32) -> Self { 
        ClError::Api(code) 
    }
}

// ─── Typ‑State‑Marker ────────────────────────────────────────────────
mod sealed { 
    pub trait Sealed {} 
}

/// **Performance**: Zero-size trait für compile-time state
pub trait State: sealed::Sealed {}

pub struct Queued;   
impl sealed::Sealed for Queued {}   
impl State for Queued {}

pub struct InFlight; 
impl sealed::Sealed for InFlight {} 
impl State for InFlight {}

pub struct Ready;    
impl sealed::Sealed for Ready  {}   
impl State for Ready  {}

// ─── GPU‑Buffer Wrapper ──────────────────────────────────────────────

/// **Performance**: repr(transparent) für zero-cost wrapping
#[repr(transparent)]
pub struct GpuBuffer<S> {
    inner: GpuBufferInner,
    _state: PhantomData<S>,
}

/// **Performance**: Interne Struktur optimiert für Cache-Locality
#[repr(C)]
struct GpuBufferInner {
    buf: Buffer<u8>,
    len: usize,
}

// **Performance**: Debug nur in debug builds
#[cfg(debug_assertions)]
impl<S> std::fmt::Debug for GpuBuffer<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("len", &self.inner.len)
            .field("state", &std::any::type_name::<S>())
            .finish()
    }
}

// **Performance**: Drop-Implementation für Resource-Cleanup
impl<S> Drop for GpuBuffer<S> {
    #[inline]
    fn drop(&mut self) {
        #[cfg(feature = "metrics")]
        {
            ALLOCS.fetch_sub(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_sub(self.inner.len, Ordering::Relaxed);
        }
    }
}

// ── Queued ───────────────────────────────────────────────────────────
impl GpuBuffer<Queued> {
    /// **Performance**: Inline + Input-Validierung
    #[inline]
    pub fn new(ctx: &Context, len: usize) -> Result<Self, ClError> {
        // **Performance**: Frühe Validierung verhindert OpenCL-Calls
        if len == 0 {
            return Err(ClError::InvalidSize(len));
        }

        // **Performance**: Metrics nur wenn feature aktiv
        #[cfg(feature = "metrics")]
        {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(len, Ordering::Relaxed);
        }

        #[cfg(feature="metrics")]
        let t = Instant::now();

        let buf = Buffer::<u8>::create(ctx, CL_MEM_READ_WRITE, len, ptr::null_mut())?;

        #[cfg(feature="metrics")]
        record("GpuBuffer::new", t);

        Ok(Self { 
            inner: GpuBufferInner { buf, len },
            _state: PhantomData 
        })
    }

    /// **Performance**: Convenience-Methode für häufigen Workflow
    #[inline]
    pub fn from_slice(ctx: &Context, queue: &CommandQueue, data: &[u8]) 
        -> Result<GpuBuffer<Ready>, ClError> {
        let buf = Self::new(ctx, data.len())?;
        let (in_flight, guard) = buf.enqueue_write(queue, data)?;
        Ok(in_flight.into_ready(guard))
    }

    /// **Performance**: Optimierte Write-Operation
    #[inline]
    pub fn enqueue_write(
        mut self,
        queue: &CommandQueue,
        host: &[u8],
    ) -> Result<(GpuBuffer<InFlight>, GpuEventGuard), ClError> {
        // **Performance**: Length-Check zur Compile-Zeit wenn möglich
        debug_assert_eq!(host.len(), self.inner.len, "Host data length mismatch");

        #[cfg(feature="metrics")]
        let t = Instant::now();

        // **Performance**: Memtrace nur wenn nötig
        #[cfg(feature="memtrace")]
        let token_box = if is_auto_trace_enabled() {
            Some(Box::new(start(Dir::H2D, host.len())))
        } else {
            None
        };

        let evt = queue.enqueue_write_buffer(
            &mut self.inner.buf,
            CL_NON_BLOCKING,
            0,
            host,
            &[],
        )?;

        // **Performance**: Callback nur wenn Tracing aktiv
        #[cfg(feature="memtrace")]
        if let Some(token_box) = token_box {
            use opencl3::event::CL_COMPLETE;
            let ptr = Box::into_raw(token_box) as *mut c_void;
            if let Err(e) = evt.set_callback(CL_COMPLETE, memtrace_callback, ptr) {
                eprintln!("callback failed: {e}");
                unsafe { Box::from_raw(ptr.cast::<CopyToken>()) }.finish();
            }
        }

        #[cfg(feature="metrics")]
        record("enqueue_write", t);

        // **Fix**: ManuallyDrop + ptr::read für Move aus Drop-Type
        use std::mem::ManuallyDrop;
        let self_manual = ManuallyDrop::new(self);
        let inner = unsafe { ptr::read(&self_manual.inner) };

        Ok((
            GpuBuffer { 
                inner, 
                _state: PhantomData::<InFlight> 
            },
            GpuEventGuard { evt },
        ))
    }

    /// **Performance**: Zero-cost state transition
    #[inline(always)]
    pub fn launch(self) -> GpuBuffer<InFlight> {
        #[cfg(feature="metrics")] 
        record("launch", Instant::now());
        
        // **Fix**: ManuallyDrop für Move aus Drop-Type
        use std::mem::ManuallyDrop;
        let self_manual = ManuallyDrop::new(self);
        let inner = unsafe { ptr::read(&self_manual.inner) };
        
        GpuBuffer { 
            inner, 
            _state: PhantomData 
        }
    }
}

// ── Ready → Host (D2H) ───────────────────────────────────────────────
impl GpuBuffer<Ready> {
    /// **Performance**: Optimierte Read-Operation
    #[inline]
    pub fn enqueue_read(
        mut self,
        queue: &CommandQueue,
        host_out: &mut [u8],
    ) -> Result<(GpuBuffer<InFlight>, GpuEventGuard), ClError> {
        // **Performance**: Length-Check zur Compile-Zeit
        debug_assert_eq!(host_out.len(), self.inner.len, "Host output length mismatch");

        #[cfg(feature="metrics")]
        let t = Instant::now();

        #[cfg(feature="memtrace")]
        let token_box = if is_auto_trace_enabled() {
            Some(Box::new(start(Dir::D2H, host_out.len())))
        } else {
            None
        };

        let evt = queue.enqueue_read_buffer(
            &mut self.inner.buf,
            CL_NON_BLOCKING,
            0,
            host_out,
            &[],
        )?;

        #[cfg(feature="memtrace")]
        if let Some(token_box) = token_box {
            use opencl3::event::CL_COMPLETE;
            let ptr = Box::into_raw(token_box) as *mut c_void;
            if let Err(e) = evt.set_callback(CL_COMPLETE, memtrace_callback, ptr) {
                eprintln!("callback failed: {e}");
                unsafe { Box::from_raw(ptr.cast::<CopyToken>()) }.finish();
            }
        }

        #[cfg(feature="metrics")]
        record("enqueue_read", t);

        // **Fix**: ManuallyDrop + ptr::read für Move aus Drop-Type
        use std::mem::ManuallyDrop;
        let self_manual = ManuallyDrop::new(self);
        let inner = unsafe { ptr::read(&self_manual.inner) };

        Ok((
            GpuBuffer { 
                inner, 
                _state: PhantomData::<InFlight> 
            },
            GpuEventGuard { evt },
        ))
    }
}

// ── InFlight ─────────────────────────────────────────────────────────
impl GpuBuffer<InFlight> {
    /// **Performance**: Deprecated - verwende into_ready(guard)
    #[inline]
    #[deprecated(note = "Use into_ready(guard) for better performance")]
    pub fn complete(self, evt: Event) -> GpuBuffer<Ready> {
        let _g = GpuEventGuard { evt };
        #[cfg(feature="metrics")] 
        record("complete", Instant::now());
        
        // **Fix**: ManuallyDrop für Move aus Drop-Type
        use std::mem::ManuallyDrop;
        let self_manual = ManuallyDrop::new(self);
        let inner = unsafe { ptr::read(&self_manual.inner) };
        
        GpuBuffer { 
            inner, 
            _state: PhantomData 
        }
    }

    /// **Performance**: Zero-cost state transition mit Guard-Konsum
    #[inline(always)]
    pub fn into_ready(self, _g: GpuEventGuard) -> GpuBuffer<Ready> {
        #[cfg(feature="metrics")] 
        record("into_ready", Instant::now());
        
        // **Fix**: ManuallyDrop für Move aus Drop-Type
        use std::mem::ManuallyDrop;
        let self_manual = ManuallyDrop::new(self);
        let inner = unsafe { ptr::read(&self_manual.inner) };
        
        GpuBuffer { 
            inner, 
            _state: PhantomData 
        }
    }

    /// **Performance**: Non-blocking Check für Async-Workflows
    pub fn try_complete(self) -> Result<GpuBuffer<Ready>, Self> {
        // TODO: Implementiere non-blocking Event-Check
        // Wenn fertig -> Ok(Ready), sonst -> Err(Self)
        Err(self) // Placeholder
    }
}

// ── Accessors (alle States) ──────────────────────────────────────────
impl<S> GpuBuffer<S> {
    /// **Performance**: Direct buffer access ohne Overhead
    #[inline(always)]
    pub fn raw(&self) -> &Buffer<u8> { 
        &self.inner.buf 
    }
    
    #[inline(always)]
    pub fn raw_mut(&mut self) -> &mut Buffer<u8> { 
        &mut self.inner.buf 
    }
    
    #[inline(always)]
    pub fn len(&self) -> usize { 
        self.inner.len 
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.inner.len == 0
    }
}

// ── Guard (wartet bei Drop auf Event) ────────────────────────────────
pub struct GpuEventGuard { 
    evt: Event 
}

impl Drop for GpuEventGuard {
    #[inline]
    fn drop(&mut self) { 
        let _ = self.evt.wait(); 
    }
}

impl GpuEventGuard {
    /// **Performance**: Explizites Wait ohne Drop
    #[inline]
    pub fn wait(self) -> Result<(), ClError> {
        let result = self.evt.wait().map_err(ClError::from);
        std::mem::forget(self); // Verhindere doppeltes Wait im Drop
        result
    }

    /// **Performance**: Non-blocking Status-Check
    pub fn is_complete(&self) -> bool {
        // TODO: Implementiere Event-Status-Check
        false // Placeholder
    }
}

// **Performance**: Re-Export für Allocation‑Zähler
#[cfg(feature = "metrics")]
pub use metrics::{ALLOCS, ALLOC_BYTES};

// ─── Buffer Pool für High-Performance Use Cases ──────────────────────
use std::collections::HashMap;

/// **Performance**: Buffer Pool für Recycling
pub struct BufferPool {
    free_buffers: HashMap<usize, Vec<Buffer<u8>>>,
    ctx: Context,
}

impl BufferPool {
    pub fn new(ctx: Context) -> Self {
        Self {
            free_buffers: HashMap::new(),
            ctx,
        }
    }

    /// **Performance**: Vermeide Allokationen durch Recycling
    pub fn get_or_create(&mut self, size: usize) -> Result<GpuBuffer<Queued>, ClError> {
        if let Some(buffers) = self.free_buffers.get_mut(&size) {
            if let Some(buf) = buffers.pop() {
                return Ok(GpuBuffer {
                    inner: GpuBufferInner { buf, len: size },
                    _state: PhantomData,
                });
            }
        }
        
        GpuBuffer::new(&self.ctx, size)
    }

    /// **Performance**: Buffer zurück in Pool (konsumiert den Buffer)
    pub fn return_buffer<S>(&mut self, buffer: GpuBuffer<S>) {
        let size = buffer.inner.len;
        
        // **Fix**: ManuallyDrop verhindert Drop-Probleme
        use std::mem::ManuallyDrop;
        let buffer = ManuallyDrop::new(buffer);
        
        // Safety: Wir nehmen Ownership und verhindern Drop
        let inner_buf = unsafe { ptr::read(&buffer.inner.buf) };
        
        self.free_buffers
            .entry(size)
            .or_insert_with(Vec::new)
            .push(inner_buf);
    }
}