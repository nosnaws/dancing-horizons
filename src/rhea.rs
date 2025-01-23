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
use std::collections::HashMap;
use std::iter;

type Population = Vec<Canditate>;
type Genotype = Vec<Direction>;
#[derive(Debug, Clone)]
pub struct RHEA {
    game: Game,
    player: String,
    populations: HashMap<String, Population>,
    tournament_size: u32,
}

#[derive(Debug, Clone)]
pub struct Canditate {
    genotype: Genotype,
    average_fitness: f32,
    total_fitness_evaluations: i32,
}

const GENO_LENGTH: usize = 5;
// const GENO_LENGTH_I32: i32 = 20;
const MUTATION_CHANCE: f32 = 0.3;
const ENEMY_SNAKE_POP_SIZE: usize = 10;
const POP_SIZE: usize = 20;

impl RHEA {
    pub fn create(game: Game, player: String) -> Self {
        let tournament_size = 5;
        let populations = HashMap::new();

        Self {
            game,
            player,
            populations,
            tournament_size,
        }
    }

    pub fn get_move(&self) -> Direction {
        let best = self.populations.get(&self.player).unwrap().get(0).unwrap();
        return best.genotype.get(0).unwrap().clone();
    }

    pub fn update_game(&self, game: Game) -> Self {
        let mut rng = SmallRng::from_entropy();
        let mut new_pops = HashMap::new();
        for (id, pop) in &self.populations {
            let updated_pop = pop
                .iter()
                .map(|c| {
                    let (_, rest) = c.genotype.split_at(1);

                    let mut new_geno = rest.to_vec();
                    new_geno.push(rng.gen());

                    let new_cand = create_candidate(new_geno);

                    return new_cand;
                })
                .collect();

            new_pops.insert(id.clone(), updated_pop);
        }

        Self {
            game,
            populations: self.evaluate_and_sort(new_pops),
            ..self.clone()
        }
    }

    pub fn evolve(&self) -> Self {
        let mut populations = self.populations.clone();
        if populations.len() == 0 {
            let mut new_pops = HashMap::new();
            for snake in &self.game.snakes {
                let l = if snake.id == *self.player {
                    POP_SIZE
                } else {
                    ENEMY_SNAKE_POP_SIZE
                };
                new_pops.insert(snake.id.clone(), create_population(l, GENO_LENGTH));
            }

            populations = self.evaluate_and_sort(new_pops);
        }

        let player_pop = populations
            .get(&self.player)
            .expect("Could not find player?");
        println!("apex {:?}", player_pop.get(0).unwrap());

        let mut rng = SmallRng::from_entropy();
        let mut new_pops = HashMap::new();
        for (id, pop) in populations {
            let apex = pop.get(0).unwrap();
            let pop_size = if *id == *self.player {
                POP_SIZE
            } else {
                ENEMY_SNAKE_POP_SIZE
            };

            let new_pop: Population = (0..pop_size)
                //  create new population via tournament selection & crossover
                .map(|_| {
                    let p1 = self.select_parent(&mut rng, &pop);
                    let p2 = self.select_parent(&mut rng, &pop);
                    // println!("parent 1 {:?}", p1);
                    // println!("parent 2 {:?}", p2);

                    return self.crossover(p1, p2);
                })
                // Leave room for the "apex"
                .take(pop_size - 1)
                //  mutate new population
                .map(|c| self.mutate(&c))
                .chain(iter::once(create_candidate(apex.genotype.clone())))
                // .map(|c| self.maybe_fix_bad_moves(&c))
                //  calculate fitness of population (& sort?)
                .collect();

            new_pops.insert(id.clone(), new_pop);
        }

        Self {
            populations: self.evaluate_and_sort(new_pops),
            ..self.clone()
        }
    }

    // pub fn get_move() -> Direction
    //  take first move of best individual

    fn mutate(&self, candidate: &Canditate) -> Canditate {
        let mut rng = SmallRng::from_entropy();

        let mut mut_chances = [0f32; GENO_LENGTH];
        rng.fill(&mut mut_chances);

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

    fn crossover(&self, c1: &Canditate, c2: &Canditate) -> Canditate {
        // probably want this
        // to be fixed length array
        let mut rng = SmallRng::from_entropy();

        let mut offspring_geno = vec![];
        for i in 0..GENO_LENGTH {
            let b: bool = rng.gen();

            let gene = if b { c1.genotype[i] } else { c2.genotype[i] };
            offspring_geno.push(gene);
        }

        return create_candidate(offspring_geno);
    }

    fn select_parent<'a>(&self, rng: &mut SmallRng, pop: &'a Population) -> &'a Canditate {
        let pop_length = pop.len();
        // randomly choose N candidates
        let mut tournament: Vec<&Canditate> = (0..self.tournament_size)
            .map(|_| pop.get(get_random_index(rng, pop_length)).unwrap())
            .collect();

        // Using an unstable sort to add some more randomness into the process
        tournament
            .sort_unstable_by(|a, b| b.average_fitness.partial_cmp(&a.average_fitness).unwrap());

        // select the one with the best fitness
        return tournament.get(0).unwrap();
    }

    fn evaluate_and_sort(
        &self,
        populations: HashMap<String, Population>,
    ) -> HashMap<String, Population> {
        let mut rng = SmallRng::from_entropy();
        let player_population = populations.get(&self.player).unwrap();
        let mut new_populations = populations.clone();

        for (i, player_candidate) in player_population.iter().enumerate() {
            let mut players: HashMap<String, (usize, Canditate)> = HashMap::new();
            players.insert(self.player.clone(), (i, player_candidate.clone()));

            for (id, cands) in &populations {
                if *id == *self.player {
                    continue;
                }

                let candidate_idx = rng.gen_range(0..ENEMY_SNAKE_POP_SIZE);
                let enemy_cand = cands[candidate_idx].clone();
                players.insert(id.clone(), (candidate_idx, enemy_cand));
            }

            let score = fitness(&self.game, &self.player, &players);

            for (id, (idx, mut cand)) in players {
                let mut new_pop = new_populations.get(&id).unwrap().clone();

                cand.total_fitness_evaluations += 1;

                let adjusted_score = if *id == *self.player { score } else { -score };

                cand.average_fitness = (((cand.total_fitness_evaluations - 1) as f32
                    * cand.average_fitness)
                    + adjusted_score)
                    / cand.total_fitness_evaluations as f32;

                new_pop[idx] = cand;
                new_populations.insert(id, new_pop);
            }
        }

        for (_, pop) in &mut new_populations {
            pop.sort_unstable_by(|a, b| b.average_fitness.partial_cmp(&a.average_fitness).unwrap());
        }
        println!("{:?}", new_populations.get(&self.player).unwrap());

        return new_populations;
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

// pub fn score_population(g: &Game, pop: &mut Population, player: &String) {
//     for cand in pop {
//         cand.fitness = fitness(g, player, &cand);
//     }
// }

pub fn fitness(
    game: &Game,
    player: &String,
    candidates: &HashMap<String, (usize, Canditate)>,
) -> f32 {
    let mut g = game.clone();
    // let mut score = 0.0;
    // let mut rng = SmallRng::from_entropy();

    for i in 0..GENO_LENGTH {
        let mut snake_moves = vec![];

        for (id, (_, cand)) in candidates {
            snake_moves.push((id.clone(), cand.genotype[i]));
        }

        g.advance_turn(snake_moves);
    }

    return score_game(&g, &player);
}

pub fn score_game(g: &Game, player: &String) -> f32 {
    let mut score = 0.0;
    let mut snakes_alive = 0.0;

    for snake in &g.snakes {
        if snake.id == *player {
            if snake.is_eliminated() {
                return -1000.0;
            }
            score += snake.health as f32 / 100.0;
            score += snake.length as f32 * 0.1;
        }

        if !snake.is_eliminated() {
            snakes_alive += 1.0;
        }
    }

    score += 100.0 / snakes_alive;
    // if snakes_alive == 1 {
    //     // we're the only ones alive now
    //     score += 100;
    // }

    return 100.0 / (100.0 + score);
}

pub fn create_population(size: usize, geno_len: usize) -> Vec<Canditate> {
    return (0..size)
        .map(|_| create_candidate(get_random_moves(geno_len)))
        .collect();
}

fn create_candidate(genotype: Vec<Direction>) -> Canditate {
    Canditate {
        genotype,
        average_fitness: 0.0,
        total_fitness_evaluations: 0,
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

    // #[test]
    // fn fitness_does_not_modify_game_state() {
    //     let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
    //     let g = Game::create(vec![s], vec![], 0, 11);
    //     let c = create_population(1, 3);
    //     let _score = fitness(&g, &String::from("test"), &c[0]);

    //     let s2 = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
    //     let g2 = Game::create(vec![s2], vec![], 0, 11);

    //     assert_eq!(g.empty, g2.empty);
    // }

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
