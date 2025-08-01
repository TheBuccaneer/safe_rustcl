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

// **Neu**: für die Allocation‑Zähler
#[cfg(feature = "metrics")]
use std::sync::atomic::Ordering;

// ─── Fehler‑Typ & cl_try! ─────────────────────────────────────────────
#[derive(thiserror::Error, Debug)]
pub enum ClError {
    #[error("OpenCL error code {0}")]
    Api(i32),
}

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
    fn from(err: opencl3::error_codes::ClError) -> Self { ClError::Api(err.0) }
}
impl From<i32> for ClError {
    fn from(code: i32) -> Self { ClError::Api(code) }
}

// ─── Typ‑State‑Marker ────────────────────────────────────────────────
mod sealed { pub trait Sealed {} }
pub trait State: sealed::Sealed {}

pub struct Queued;   impl sealed::Sealed for Queued {}   impl State for Queued {}
pub struct InFlight; impl sealed::Sealed for InFlight {} impl State for InFlight {}
pub struct Ready;    impl sealed::Sealed for Ready  {}   impl State for Ready  {}

// ─── GPU‑Buffer Wrapper ──────────────────────────────────────────────
pub struct GpuBuffer<S> {
    buf: Buffer<u8>,
    len: usize,
    _state: PhantomData<S>,
}

// ── Queued ───────────────────────────────────────────────────────────
impl GpuBuffer<Queued> {
    pub fn new(ctx: &Context, len: usize) -> Result<Self, ClError> {
        // **Allocation‑Zähler** (neu)
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

        Ok(Self { buf, len, _state: PhantomData })
    }

    pub fn enqueue_write(
        mut self,
        queue: &CommandQueue,
        host: &[u8],
    ) -> Result<(GpuBuffer<InFlight>, GpuEventGuard), ClError> {

        #[cfg(feature="metrics")]
        let t = Instant::now();

        #[cfg(feature="memtrace")]
        let token_box = if is_auto_trace_enabled() {
            Some(Box::new(start(Dir::H2D, host.len())))
        } else {
            None
        };

        let evt = queue.enqueue_write_buffer(
            &mut self.buf,
            CL_NON_BLOCKING,
            0,
            host,
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
        record("enqueue_write", t);

        Ok((
            GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData::<InFlight> },
            GpuEventGuard { evt },
        ))
    }

    pub fn launch(self) -> GpuBuffer<InFlight> {
        #[cfg(feature="metrics")] record("launch", Instant::now());
        GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData }
    }
}

// ── Ready → Host (D2H) ───────────────────────────────────────────────
impl GpuBuffer<Ready> {
    pub fn enqueue_read(
        mut self,
        queue: &CommandQueue,
        host_out: &mut [u8],
    ) -> Result<(GpuBuffer<InFlight>, GpuEventGuard), ClError> {

        #[cfg(feature="metrics")]
        let t = Instant::now();

        #[cfg(feature="memtrace")]
        let token_box = if is_auto_trace_enabled() {
            Some(Box::new(start(Dir::D2H, host_out.len())))
        } else {
            None
        };

        let evt = queue.enqueue_read_buffer(
            &mut self.buf,
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

        Ok((
            GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData::<InFlight> },
            GpuEventGuard { evt },
        ))
    }
}

// ── InFlight ─────────────────────────────────────────────────────────
impl GpuBuffer<InFlight> {
    pub fn complete(self, evt: Event) -> GpuBuffer<Ready> {
        let _g = GpuEventGuard { evt };
        #[cfg(feature="metrics")] record("complete", Instant::now());
        GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData }
    }

    pub fn into_ready(self, _g: GpuEventGuard) -> GpuBuffer<Ready> {
        #[cfg(feature="metrics")] record("into_ready", Instant::now());
        GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData }
    }
}

// ── Accessors (alle States) ──────────────────────────────────────────
impl<S> GpuBuffer<S> {
    pub fn raw(&self) -> &Buffer<u8> { &self.buf }
    pub fn raw_mut(&mut self) -> &mut Buffer<u8> { &mut self.buf }
    pub fn len(&self) -> usize { self.len }
}

// ── Guard (wartet bei Drop auf Event) ────────────────────────────────
pub struct GpuEventGuard { evt: Event }
impl Drop for GpuEventGuard {
    fn drop(&mut self) { let _ = self.evt.wait(); }
}

// **Neu**: Re-Export für Allocation‑Zähler
#[cfg(feature = "metrics")]
pub use metrics::{ALLOCS, ALLOC_BYTES};