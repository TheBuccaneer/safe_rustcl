use opencl3::{
    context::Context,
    memory::{Buffer, CL_MEM_READ_WRITE},
    event::{Event},
};

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
    buf: Buffer<u8>,
    len: usize,
    _state: PhantomData<S>,
}



impl GpuBuffer<Queued> {
    /// legt ein neues GPU‑Buffer an, noch **nicht** synchronisiert
    pub fn new(context: &Context, len: usize) -> Result<Self, ClError> {
        let buf = Buffer::<u8>::create(
            context,
            CL_MEM_READ_WRITE,
            len,
            ptr::null_mut(),
        )?;
        Ok(Self { buf, len, _state: PhantomData })
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
    evt.wait()?;   // nutzt das OpenCL3‑Wrapper‑safe wait
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
}


pub struct GpuEventGuard {
    evt: opencl3::event::Event,
}
impl Drop for GpuEventGuard {
    fn drop(&mut self) {
        let _ = self.evt.wait();  // blockiert bis Kernel fertig
    }
}