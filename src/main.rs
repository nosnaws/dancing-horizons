pub mod bitboard;
pub mod rhea;

use crate::bitboard::{Direction, Game, Move, Snake};
use crate::rhea::RHEA;

fn main() {
    let s = Snake::create(String::from("test1"), 100, vec![0, 1, 2]);
    // let s2 = Snake::create(String::from("test2"), 100, vec![10, 9, 8]);

    let g = Game::create(vec![s], vec![], 0, 11);

    let mut pop = RHEA::create_population(3, 2);
    RHEA::score_population(&g, &mut pop, &String::from("test1"));
    // let mut r = RHEA::create(g, String::from("test1"));
    

    // println!("{:?}", r);
    println!("{:?}", pop);
    // println!("{:?}", fit);
}
