use crate::bitboard::Snake;

pub mod bitboard;

fn main() {
    let mut snake = Snake::create(String::from("dancing-horizon"), 100, vec![0, 1, 2]);

    assert_eq!(snake.head_board, 1);

    println!("{:?}!\nLength: {}", snake, snake.length());

    snake.move_in_dir(&bitboard::Direction::Up, 11);

    assert_eq!(snake.head_board, 1 << 11);

    println!("{:?}!\nLength: {}", snake, snake.length())
}
