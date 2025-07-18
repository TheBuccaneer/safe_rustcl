pub struct Queued;
pub struct Ready;

pub struct GpuBuffer<S> {
    mem: opencl3::memory::Buffer::<u8>,
    _state: std::marker::PhantomData<S>,
}