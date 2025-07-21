#[cfg(feature = "metrics")]
mod metrics;

#[cfg(feature = "metrics")]
pub use metrics::*;

pub mod memtracer;

use opencl3::{
    context::Context,
    memory::{Buffer, CL_MEM_READ_WRITE},
    event::Event,
    command_queue::CommandQueue,

};
use opencl3::types::CL_NON_BLOCKING;


use std::{marker::PhantomData, ptr};

mod sealed { pub trait Sealed {} }
pub trait State: sealed::Sealed {}

pub struct Queued;  impl sealed::Sealed for Queued {}  impl State for Queued {}
pub struct InFlight;impl sealed::Sealed for InFlight {}impl State for InFlight {}
pub struct Ready;   impl sealed::Sealed for Ready  {}  impl State for Ready  {}

#[derive(thiserror::Error, Debug)]
pub enum ClError {
    #[error("OpenCL error code {0}")]
    Api(i32),
}

/// Makro: wandelt cl_int in Result
/*#[warn(unused_macros)]
macro_rules! cl_try {
    ($expr:expr) => {
        let err = unsafe { $expr };
        if err != 0 {
            return Err(crate::ClError::Api(err));
        }
    };
}
pub(crate) use cl_try;
*/

impl From<opencl3::error_codes::ClError> for ClError {
    fn from(err: opencl3::error_codes::ClError) -> Self {
        ClError::Api(err.0)
    }
}

impl From<i32> for ClError {
    fn from(code: i32) -> Self {
        ClError::Api(code)
    }
}



pub struct GpuBuffer<S> {
    pub buf: Buffer<u8>,
    pub len: usize,
    pub _state: PhantomData<S>,
}



impl GpuBuffer<Queued> {
    /// legt ein neues GPU‑Buffer an, noch **nicht** synchronisiert
    pub fn new(context: &Context, len: usize) -> Result<Self, ClError> {
        #[cfg(feature = "metrics")]
        let t0 = std::time::Instant::now();

        let buf = Buffer::<u8>::create(
            context,
            CL_MEM_READ_WRITE,
            len,
            ptr::null_mut(),
        )?;

        #[cfg(feature = "metrics")]
        record("GpuBuffer::new", t0);

        Ok(Self { buf, len, _state: PhantomData })
    }

    pub fn enqueue_write(
    mut self,
    queue: &CommandQueue,
    host_data: &[u8],
) -> Result<(GpuBuffer<InFlight>, GpuEventGuard), ClError> {
    let evt = queue.enqueue_write_buffer(
        &mut self.buf,
        CL_NON_BLOCKING,
        0,
        host_data,
        &[],
    )?;
    Ok((
        GpuBuffer {
            buf: self.buf,
            len: self.len,
            _state: PhantomData::<InFlight>,
        },
        GpuEventGuard { evt },
    ))
}

    
    /// wartet auf GPU‑Fertigstellung und überführt in Ready‑State
    //pub fn into_ready(self, evt: cl_event) -> Result<GpuBuffer<Ready>, ClError> {
      //  wait_for_events(&[evt])?;          // ⬅️ hier liegt die eigentliche Synchronisation
        //Ok(GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData })
    //}

   /* pub fn into_ready(self, evt: cl_event) -> Result<GpuBuffer<Ready>, ClError> {
    // Nur warten, wenn evt != NULL
    if !evt.is_null() {
        wait_for_events(&[evt])?;
    }
    Ok(GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData })
    */

    pub fn into_ready(self, evt: Event) -> Result<GpuBuffer<Ready>, ClError> {
        #[cfg(feature = "metrics")]
        let t1 = std::time::Instant::now();

        evt.wait()?;

        #[cfg(feature = "metrics")]
        record("into_ready", t1);

        Ok(GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData })
    }


}




impl<S> GpuBuffer<S> {
    /// Zugriff auf die interne OpenCL Buffer-Referenz
    pub fn raw(&self) -> &Buffer<u8> {
        &self.buf
    }

    /// Länge des Buffers in Bytes
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn raw_mut(&mut self) -> &mut Buffer<u8> {
        &mut self.buf
    }
}

impl GpuBuffer<Queued> {
    pub fn launch(self) -> GpuBuffer<InFlight> {
        GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData }
    }
}

impl GpuBuffer<InFlight> {
    pub fn complete(self, evt: opencl3::event::Event) -> GpuBuffer<Ready> {
        let _guard = GpuEventGuard { evt };
        GpuBuffer { buf: self.buf, len: self.len, _state: PhantomData }
    }

    pub fn into_ready(self, _guard: GpuEventGuard) -> GpuBuffer<Ready> {
        // Der Drop von _guard wartet bis Event fertig ist
        GpuBuffer {
            buf: self.buf,
            len: self.len,
            _state: PhantomData::<Ready>,
        }
    }

}


pub struct GpuEventGuard {
    evt: opencl3::event::Event,
}
impl Drop for GpuEventGuard {
    fn drop(&mut self) {
        let _ = self.evt.wait();  // blockiert bis Kernel fertig
    }
}