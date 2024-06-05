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
use rand::rngs::ThreadRng;
use rand::Rng;
use std::iter;

#[derive(Debug, Clone)]
pub struct RHEA {
    game: Game,
    player: String,
    pop: Vec<Canditate>,
    mutation_chance: f32,
    crossover_chance: f32,
    geno_length: usize,
    tournament_size: u32,
    population_size: usize,
}

type Population = Vec<Canditate>;
#[derive(Debug, Clone)]
pub struct Canditate {
    genotype: Vec<Direction>,
    fitness: i32,
}

impl RHEA {
    pub fn create(game: Game, player: String) -> Self {
        let mutation_chance = 0.2;
        let crossover_chance = 0.9;
        let geno_length = 3;
        let population_size = 25;
        let tournament_size = 10;
        let mut pop: Population = create_population(population_size, geno_length)
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
            mutation_chance,
            crossover_chance,
            geno_length,
            tournament_size,
            population_size,
        }
    }

    pub fn get_move(&self) -> Direction {
        let best = self.pop.get(0).unwrap();
        return best.genotype.get(0).unwrap().clone();
    }

    // pub fn update_game(&self, game: Game) -> Self {
    // }

    pub fn evolve(&self) -> Self {
        println!("apex {:?}", self.pop.get(0).unwrap());
        let apex = create_candidate(self.pop.get(0).unwrap().genotype.clone());

        let pairs = (self.population_size / 2) + (self.population_size % 2);
        let mut new_pop: Population = (0..pairs)
            //  create new population via tournament selection & crossover
            .flat_map(|_| {
                let p1 = self.select_parent(&self.pop);
                let p2 = self.select_parent(&self.pop);

                return self.crossover(p1, p2);
            })
            // Leave room for the "apex"
            .take(self.population_size - 1)
            .chain(iter::once(apex))
            //  mutate new population
            .map(|c| self.mutate(&c))
            //  calculate fitness of population (& sort?)
            .map(|c| Canditate {
                fitness: fitness(&self.game, &self.player, &c),
                ..c
            })
            .collect();

        // println!("{:?}", new_pop);

        new_pop.sort_unstable_by(|a, b| b.fitness.cmp(&a.fitness));

        // println!("{:?}", new_pop);

        //  return with new population
        Self {
            pop: new_pop,
            ..self.clone()
        }
    }

    // pub fn get_move() -> Direction
    //  take first move of best individual

    fn mutate(&self, candidate: &Canditate) -> Canditate {
        let mut rng = rand::thread_rng();

        let m: f32 = rng.gen();

        if m < 1.0 - self.mutation_chance {
            return create_candidate(candidate.genotype.clone());
        }

        let i: usize = get_random_index(self.geno_length);
        let new_move = get_random_move();

        let mut geno = candidate.genotype.clone();

        geno[i] = new_move;

        return create_candidate(geno);
    }

    fn crossover(&self, c1: &Canditate, c2: &Canditate) -> Vec<Canditate> {
        // probably want this
        // to be fixed length array
        let mut rng = rand::thread_rng();

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
        let cross_over_point = get_random_index(self.geno_length);
        // create 2 new candidates with each half of parent's genotype at crossover point
        let c1_pair = c1.genotype.split_at(cross_over_point);
        let c2_pair = c2.genotype.split_at(cross_over_point);

        return vec![
            create_candidate([c1_pair.0, c2_pair.1].concat()),
            create_candidate([c2_pair.0, c1_pair.1].concat()),
        ];
    }

    fn select_parent<'a>(&self, pop: &'a Population) -> &'a Canditate {
        let pop_length = self.pop.len();
        // randomly choose N candidates
        let mut tournament: Vec<&Canditate> = (0..self.tournament_size)
            .map(|_| pop.get(get_random_index(pop_length)).unwrap())
            .collect();

        // Using an unstable sort to add some more randomness into the process
        tournament.sort_unstable_by(|a, b| a.fitness.cmp(&b.fitness));

        // select the one with the best fitness
        return tournament.get(0).unwrap();
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
        if snake.id == *player && snake.is_eliminated() {
            return 0;
        } else {
            score += snake.health;
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

pub fn get_random_moves(n: usize) -> Vec<Direction> {
    let mut moves = vec![];

    for _i in 0..n {
        moves.push(get_random_move());
    }

    return moves;
}

pub fn get_random_move() -> Direction {
    rand::random()
}

fn get_random_index(len: usize) -> usize {
    let mut rng = rand::thread_rng();
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
