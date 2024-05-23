// Rolling Horizon Evolutionary Algorithm
// for snakes!

// TODO:
// - "fix" bad moves in candidates
// - cross over function (uniform)
// - mutation function
// - tournament selection
// - apex selection
// - allow starting with an initial population

use crate::bitboard::{Direction, Game};

#[derive(Debug)]
pub struct RHEA {
    game: Game,
    player: String,
    pop: Vec<Canditate>
}

type Population = Vec<Canditate>;
#[derive(Debug)]
pub struct Canditate {
    genotype: Vec<Direction>,
    fitness: i32,
}

impl RHEA {
    pub fn create(game: Game, player: String) -> Self {
        Self { game, player, pop: vec![] }
    }

    pub fn create_population(size: i32, geno_len: i32) -> Vec<Canditate> {
        let mut candidates: Vec<Canditate> = vec![];

        for _i in 0..size {
            candidates.push(Canditate {
                genotype: RHEA::get_random_moves(geno_len),
                fitness: 0,
            });
        }

        candidates
    }

    pub fn score_population(g: &Game,  pop: &mut Population, player: &String) {
       for cand in pop {
            cand.fitness = RHEA::fitness(g, player, &cand);
        }
    }

    pub fn fitness(game: &Game, player: &String, c: &Canditate) -> i32 {
        let mut g = game.clone();
        for dir in &c.genotype {
            g.advance_turn(vec![(player.clone(), dir.clone())]);
        }

        return RHEA::score_game(g, &player);
    }

    pub fn score_game(g: Game, player: &String) -> i32 {
        let mut score = 0;
        let mut snakes_alive = 0;

        for snake in g.snakes {
            if snake.id == *player && snake.is_eliminated() {
                return -10000;
            } else {
                score += snake.health;
            }

            if !snake.is_eliminated() {
                snakes_alive += 1;
            }
        }

        if snakes_alive == 1 {
            // we're the only ones alive now
            score += 10000;
        }

        return score;
    }

    pub fn get_random_moves(n: i32) -> Vec<Direction> {
        let mut moves = vec![];

        for _i in 0..n {
            moves.push(RHEA::get_random_move());
        }

        return moves;
    }

    pub fn get_random_move() -> Direction {
        rand::random()
    }
}
