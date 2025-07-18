use criterion::{Criterion, criterion_group, criterion_main};

fn hello_bench(c: &mut Criterion) {
    c.bench_function("hello_world", |b| {
        b.iter(|| {
            // Hier wird die einfachste Operation durchgeführt
            let result = 1 + 1;
            assert_eq!(result, 2);
        });
    });
}

// Diese beiden Makros sind notwendig, damit Criterion den Benchmark ausführt
criterion_group!(benches, hello_bench);
criterion_main!(benches);
