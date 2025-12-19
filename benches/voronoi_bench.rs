use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dancing_horizons::bitboard::{Game, Snake};

fn create_game_2_snakes_opposite() -> Game {
    // Two snakes in opposite corners - maximum distance
    let s1 = Snake::create(String::from("snake1"), 100, vec![0, 1, 2]);
    let s2 = Snake::create(String::from("snake2"), 100, vec![120, 119, 118]);
    Game::create(vec![s1, s2], vec![], 0, 11)
}

fn create_game_2_snakes_close() -> Game {
    // Two snakes close together - many contested cells
    let s1 = Snake::create(String::from("s1"), 100, vec![50, 49, 48]);
    let s2 = Snake::create(String::from("s2"), 100, vec![52, 53, 54]);
    Game::create(vec![s1, s2], vec![], 0, 11)
}

fn create_game_4_snakes_corners() -> Game {
    // Four snakes, one in each corner
    let s1 = Snake::create(String::from("s1"), 100, vec![0, 1, 2]);
    let s2 = Snake::create(String::from("s2"), 100, vec![10, 9, 8]);
    let s3 = Snake::create(String::from("s3"), 100, vec![110, 111, 112]);
    let s4 = Snake::create(String::from("s4"), 100, vec![120, 119, 118]);
    Game::create(vec![s1, s2, s3, s4], vec![], 0, 11)
}

fn create_game_single_snake() -> Game {
    // Single snake - should control everything
    let s = Snake::create(String::from("solo"), 100, vec![60, 59, 58]);
    Game::create(vec![s], vec![], 0, 11)
}

fn benchmark_voronoi(c: &mut Criterion) {
    c.bench_function("voronoi_2_snakes_opposite", |b| {
        let game = create_game_2_snakes_opposite();
        b.iter(|| black_box(game.calculate_voronoi_scores()));
    });

    c.bench_function("voronoi_2_snakes_close", |b| {
        let game = create_game_2_snakes_close();
        b.iter(|| black_box(game.calculate_voronoi_scores()));
    });

    c.bench_function("voronoi_4_snakes_corners", |b| {
        let game = create_game_4_snakes_corners();
        b.iter(|| black_box(game.calculate_voronoi_scores()));
    });

    c.bench_function("voronoi_single_snake", |b| {
        let game = create_game_single_snake();
        b.iter(|| black_box(game.calculate_voronoi_scores()));
    });
}

criterion_group!(benches, benchmark_voronoi);
criterion_main!(benches);
