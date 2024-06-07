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
use rand::rngs::SmallRng;
use rand::rngs::ThreadRng;
use rand::{Rng, SeedableRng};
use std::iter;

#[derive(Debug, Clone)]
pub struct RHEA {
    game: Game,
    player: String,
    pop: Vec<Canditate>,
    crossover_chance: f32,
    tournament_size: u32,
    population_size: usize,
}

type Population = Vec<Canditate>;
#[derive(Debug, Clone)]
pub struct Canditate {
    genotype: Vec<Direction>,
    fitness: i32,
}

const GENO_LENGTH: usize = 20;
const MUTATION_CHANCE: f32 = 0.3;

impl RHEA {
    pub fn create(game: Game, player: String) -> Self {
        let crossover_chance = 1.0;
        let population_size = 50;
        let tournament_size = 3;
        let mut pop: Population = create_population(population_size, GENO_LENGTH)
            .iter()
            .map(|c| Canditate {
                fitness: fitness(&game, &player, &c),
                ..c.clone()
            })
            .collect();

        pop.sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));
        Self {
            game,
            player,
            pop,
            crossover_chance,
            tournament_size,
            population_size,
        }
    }

    pub fn get_move(&self) -> Direction {
        let best = self.pop.get(0).unwrap();
        return best.genotype.get(0).unwrap().clone();
    }

    pub fn update_game(&self, game: Game) -> Self {
        let mut rng = SmallRng::from_entropy();
        let updated_pop = self
            .pop
            .iter()
            .map(|c| {
                let (_, rest) = c.genotype.split_at(1);

                let mut new_geno = rest.to_vec();
                new_geno.push(rng.gen());

                let mut new_cand = create_candidate(new_geno);
                new_cand.fitness = fitness(&game, &self.player, &new_cand);

                return new_cand;
            })
            .collect();

        Self {
            game,
            pop: updated_pop,
            ..self.clone()
        }
    }

    pub fn evolve(&self) -> Self {
        println!("apex {:?}", self.pop.get(0).unwrap());
        let apex = create_candidate(self.pop.get(0).unwrap().genotype.clone());

        let mut rng = SmallRng::from_entropy();
        let pairs = (self.population_size / 2) + (self.population_size % 2);
        let mut new_pop: Population = (0..pairs)
            //  create new population via tournament selection & crossover
            .flat_map(|_| {
                let p1 = self.select_parent(&mut rng, &self.pop);
                let p2 = self.select_parent(&mut rng, &self.pop);
                // println!("parent 1 {:?}", p1);
                // println!("parent 2 {:?}", p2);

                return self.crossover(p1, p2);
            })
            // Leave room for the "apex"
            .take(self.population_size - 1)
            //  mutate new population
            .map(|c| self.mutate(&c))
            .chain(iter::once(apex))
            // .map(|c| self.maybe_fix_bad_moves(&c))
            //  calculate fitness of population (& sort?)
            .map(|c| Canditate {
                fitness: fitness(&self.game, &self.player, &c),
                ..c
            })
            .collect();

        // println!("{:?}", new_pop);

        new_pop.sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));

        // println!("new pop {:?}", new_pop);

        //  return with new population
        Self {
            pop: new_pop,
            ..self.clone()
        }
    }

    // pub fn get_move() -> Direction
    //  take first move of best individual

    fn mutate(&self, candidate: &Canditate) -> Canditate {
        let mut rng = SmallRng::from_entropy();

        // rng.gen()
        let mut mut_chances = [0f32; GENO_LENGTH];
        rng.fill(&mut mut_chances);

        // let mut new_geno = vec![];
        // for (i, mc) in mut_chances {
        //         if mc < 1.0 - MUTATION_CHANCE {
        //             return *candidate.genotype[i];
        //         }

        //         return get_random_move();

        // }
        let mutated_geno = mut_chances
            .iter()
            .enumerate()
            .map(|(i, mc)| {
                if *mc < 1.0 - MUTATION_CHANCE {
                    return candidate.genotype[i];
                }

                return rng.gen();
            })
            .collect();

        return create_candidate(mutated_geno);
    }

    fn crossover(&self, c1: &Canditate, c2: &Canditate) -> Vec<Canditate> {
        // probably want this
        // to be fixed length array
        let mut rng = SmallRng::from_entropy();

        let m: f32 = rng.gen();

        // check for cross over, return original candidates if not
        if m < 1.0 - self.crossover_chance {
            vec![
                create_candidate(c1.genotype.clone()),
                create_candidate(c2.genotype.clone()),
            ];
        }
        // generate random crossover point
        // TODO: Cross over has the chance to happen at the first or last index
        // of the genotype, effectively not doing a crossover. Might want to
        // eliminate that chance by only generating random numbers between 1 and length - 1.
        let cross_over_point = get_random_index(&mut rng, GENO_LENGTH);
        // create 2 new candidates with each half of parent's genotype at crossover point
        let c1_pair = c1.genotype.split_at(cross_over_point);
        let c2_pair = c2.genotype.split_at(cross_over_point);

        return vec![
            create_candidate([c1_pair.0, c2_pair.1].concat()),
            create_candidate([c2_pair.0, c1_pair.1].concat()),
        ];
    }

    fn select_parent<'a>(&self, rng: &mut SmallRng, pop: &'a Population) -> &'a Canditate {
        let pop_length = self.pop.len();
        // randomly choose N candidates
        let mut tournament: Vec<&Canditate> = (0..self.tournament_size)
            .map(|_| pop.get(get_random_index(rng, pop_length)).unwrap())
            .collect();

        // Using an unstable sort to add some more randomness into the process
        tournament.sort_unstable_by(|a, b| a.fitness.cmp(&b.fitness));

        // select the one with the best fitness
        return tournament.get(0).unwrap();
    }

    // fn maybe_fix_bad_moves(&self, c: &Canditate) -> Canditate {
    //     let mut geno = c.genotype.clone();
    //     for i in 0..GENO_LENGTH {
    //         let first_move = geno[i];
    //         if i + 1 >= GENO_LENGTH {
    //             break;
    //         }
    //         let second_move = geno[i + 1];

    //         if is_bad_move(first_move, second_move) {
    //             let n = get_random_move();
    //             geno[i + 1] = n
    //         }
    //     }

    //     return create_candidate(geno);
    // }
}

// returns true if the second argument is a bad move
fn is_bad_move(m1: Direction, m2: Direction) -> bool {
    match m1 {
        Direction::Up => match m2 {
            Direction::Down => true,
            _ => false,
        },
        Direction::Down => match m2 {
            Direction::Up => true,
            _ => false,
        },
        Direction::Left => match m2 {
            Direction::Right => true,
            _ => false,
        },
        Direction::Right => match m2 {
            Direction::Left => true,
            _ => false,
        },
    }
}

pub fn score_population(g: &Game, pop: &mut Population, player: &String) {
    for cand in pop {
        cand.fitness = fitness(g, player, &cand);
    }
}

pub fn fitness(game: &Game, player: &String, c: &Canditate) -> i32 {
    let mut g = game.clone();
    let mut score = 0;
    for dir in &c.genotype {
        g.advance_turn(vec![(player.clone(), dir.clone())]);
        score += score_game(&g, &player);
    }

    return score;
}

pub fn score_game(g: &Game, player: &String) -> i32 {
    let mut score = 0;
    let mut snakes_alive = 0;

    for snake in &g.snakes {
        if snake.id == *player {
            if snake.is_eliminated() {
                return -1000;
            }
            // score += snake.health;
        }

        if !snake.is_eliminated() {
            snakes_alive += 1;
        }
    }

    if snakes_alive == 1 {
        // we're the only ones alive now
        score += 100;
    }

    return score;
}

pub fn create_population(size: usize, geno_len: usize) -> Vec<Canditate> {
    return (0..size)
        .map(|_| create_candidate(get_random_moves(geno_len)))
        .collect();
}

fn create_candidate(genotype: Vec<Direction>) -> Canditate {
    Canditate {
        genotype,
        fitness: 0,
    }
}

fn get_random_moves(n: usize) -> Vec<Direction> {
    let mut moves = vec![];
    let mut rng = SmallRng::from_entropy();

    for _i in 0..n {
        moves.push(rng.gen());
    }

    return moves;
}


fn get_random_index(rng: &mut SmallRng, len: usize) -> usize {
    rng.gen_range(0..len).try_into().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard::{Game, Snake};

    #[test]
    fn fitness_does_not_modify_game_state() {
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let c = create_population(1, 3);
        let _score = fitness(&g, &String::from("test"), &c[0]);

        let s2 = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g2 = Game::create(vec![s2], vec![], 0, 11);

        assert_eq!(g.empty, g2.empty);
    }

    #[test]
    fn mutates() {
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let c = create_population(1, 3);
        let geno_copy = c[0].genotype.clone();
        let r = RHEA::create(g, String::from("test"));
        let cm = r.mutate(&c[0]);

        assert_eq!(cm.genotype, geno_copy);
    }
}
