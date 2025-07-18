use opencl3::{
    command_queue::{CommandQueue},
    context::Context,
    memory::{Buffer, CL_MEM_READ_WRITE},
    event::{wait_for_events, Event},
    types::{cl_event, CL_NON_BLOCKING, CL_BLOCKING},
};

use std::{marker::PhantomData, ptr};



pub struct Queued;
pub struct Ready;

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
    pub fn into_ready(self, evt: cl_event) -> Result<GpuBuffer<Ready>, ClError> {
        wait_for_events(&[evt])?;          // ⬅️ hier liegt die eigentliche Synchronisation
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
}
