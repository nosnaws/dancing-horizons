use rand::{
    distributions::{Distribution, Standard},
    seq::SliceRandom,
    Rng,
};
// BOARD (11x11, indices 0-120)
// 110 | 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120
// 099 | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109
// 088 | 089 | 090 | 091 | 092 | 093 | 094 | 095 | 096 | 097 | 098
// 077 | 078 | 079 | 080 | 081 | 082 | 083 | 084 | 085 | 086 | 087
// 066 | 067 | 068 | 069 | 070 | 071 | 072 | 073 | 074 | 075 | 076
// 055 | 056 | 057 | 058 | 059 | 060 | 061 | 062 | 063 | 064 | 065
// 044 | 045 | 046 | 047 | 048 | 049 | 050 | 051 | 052 | 053 | 054
// 033 | 034 | 035 | 036 | 037 | 038 | 039 | 040 | 041 | 042 | 043
// 022 | 023 | 024 | 025 | 026 | 027 | 028 | 029 | 030 | 031 | 032
// 011 | 012 | 013 | 014 | 015 | 016 | 017 | 018 | 019 | 020 | 021
// 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010
pub type Board = u128;

pub const WIDTH: u128 = 11;
pub const MAX_HEALTH: i32 = 100;

const LEFT_MASK: u128 = 
0b0000000_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001;

const RIGHT_MASK: u128 = 
0b0000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000;

const TOP_MASK: u128 = 
0b0000000_11111111111_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000;

const BOTTOM_MASK: u128 = 
0b0000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_11111111111;

#[derive(Debug, Clone)]
pub struct Snake {
    pub id: String,
    pub health: i32,
    pub body: Vec<u128>,
    pub head_board: Board,
    pub body_board: Board,
    pub length: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Distribution<Direction> for Standard {
 fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Direction{
        // match rng.gen_range(0, 3) { // rand 0.5, 0.6, 0.7
        match rng.gen_range(0..=3) { // rand 0.8
            0 => Direction::Up,
            1 => Direction::Left,
            2 => Direction::Right,
            _ => Direction::Down,
        }
    }
}



impl Snake {
    pub fn create(id: String, health: i32, body: Vec<u128>) -> Self {
        let head_board = set_index(0, body[0], 1);
        // Not sure I want the clone here, need to read up on that
        let body_board = body
            .clone()
            .into_iter()
            .fold(0, |acc, p| set_index(acc, p, 1));

        let length = body.len();

        Self {
            id,
            health,
            body,
            head_board,
            body_board,
            length,
        }
    }

    pub fn length(&self) -> usize {
        self.length
    }
    pub fn is_eliminated(&self) -> bool {
        self.health < 1
    }

    fn head(&self) -> u128 {
        self.body[0]
    }

    pub fn feed(&mut self) {
        self.health = MAX_HEALTH;
        let tail = self.tail();

        self.body.push(tail);
    }

    fn tail(&self) -> u128 {
        self.body.last().unwrap().clone()
    }

    fn just_ate(&self) -> bool {
        let len = self.body.len();
        self.body[len-2] == self.body[len-1]
    }

    pub fn move_tail(&mut self) {
        let tail = self.body.pop().unwrap();
        let len = self.body.len();
        let just_ate = tail == self.body[len-1];
        if !just_ate {
            self.body_board = set_index(self.body_board, tail, 0);
        }
    }

    pub fn move_head(&mut self, dir: &Direction, width: u128) {
        let new_head_b = {
            match dir {
                Direction::Left => self.head_board >> 1,
                Direction::Right => self.head_board << 1,
                Direction::Up => self.head_board << width,
                Direction::Down => self.head_board >> width,
            }
        };

        let new_body_b = self.body_board | self.head_board;

        self.head_board = new_head_b;
        self.body_board = new_body_b;

        let new_head = dir_to_index(self.head(), dir, width);
        // add head to beginning of body vector
        self.body.insert(0, new_head);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_snake() {
        let s = Snake::create(String::from("test"), 100, vec![0,1,2]);

        assert_eq!(s.id, "test");
        assert_eq!(s.health, 100);
        assert_eq!(s.body, vec![0,1,2]);
        assert_eq!(s.head_board, 1);
        assert_eq!(s.body_board, 7);
    }

    #[test]
    fn length() {
        let s = Snake::create(String::from("test"), 100, vec![0,1,2]);
        assert_eq!(s.length(), 3);
    }

    #[test]
    fn head() {
        let s = Snake::create(String::from("test"), 100, vec![0,1,2]);
        assert_eq!(s.head(), 0);
    }

    #[test]
    fn feed() {
        let mut s = Snake::create(String::from("test"), 50, vec![0,1,2]);
        s.feed();
        assert_eq!(s.health, 100);
        assert_eq!(s.body, vec![0,1,2,2]);
    }

    #[test]
    fn move_up() {
        let mut s = Snake::create(String::from("test"), 100, vec![0,1,2]);

        s.move_tail();
        s.move_head(&super::Direction::Up, 11);

        assert_eq!(s.body, vec![11,0,1]);
        // assert_eq!(s.head_board, 2048);
        dbg!(s.body);
        assert_eq!(s.head_board, 0b100000000000);
        assert_eq!(s.body_board, 0b000000000011);
    }

    #[test]
    fn basic_move() {
        let s = Snake::create(String::from("test"), 100, vec![0,1,2]);

        let mut g = Game::create(vec![s], vec![], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Up)]);
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 99);
        assert_eq!(s.body, vec![11,0,1]);
        assert_eq!(s.head_board, 0b100000000000);
        assert_eq!(s.body_board, 0b000000000011);
    }

    #[test]
    fn eat_food() {
        let s = Snake::create(String::from("test"), 100, vec![0,1,2]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Up)]);
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 100);
        assert_eq!(s.body, vec![11,0,1,1]);
        assert_eq!(g.food, 0);
    }

    #[test]
    fn out_of_bounds_left() {
        let s = Snake::create(String::from("test"), 100, vec![0,1,2]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Left)]);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
        // When all snakes are eliminated, occupied should be 0 (no living snakes)
        assert_eq!(g.occupied, 0);
    }

    #[test]
    fn out_of_bounds_bottom() {
        let s = Snake::create(String::from("test"), 100, vec![0,1,2]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Down)]);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
        assert_eq!(g.occupied, 0);
    }

    #[test]
    fn out_of_bounds_right() {
        let s = Snake::create(String::from("test"), 100, vec![10,9,8]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Right)]);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
        assert_eq!(g.occupied, 0);
    }

    #[test]
    fn out_of_bounds_top() {
        let s = Snake::create(String::from("test"), 100, vec![111,112,113]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Up)]);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
        assert_eq!(g.occupied, 0);
    }

    #[test]
    fn self_collision() {
        let s = Snake::create(String::from("test"), 100, vec![12, 11,0, 1,2]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Down)]);
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
    }

    #[test]
    fn move_back_on_neck() {
        let s = Snake::create(String::from("test"), 100, vec![0, 1,2]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Right)]);
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
    }

    #[test]
    fn move_back_on_neck_vertical() {
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);

        let mut g = Game::create(vec![s], vec![], 0, 11);
        // g.advance_turn(vec![(String::from("test"), Direction::Up)]);
        // dbg!(&g);

        g.advance_turn(vec![(String::from("test"), Direction::Up)]);
        g.advance_turn(vec![(String::from("test"), Direction::Down)]);
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
    }


    #[test]
    fn snake_body_collision() {
        let s = Snake::create(String::from("test1"), 100, vec![0,1,2]);
        let s2 = Snake::create(String::from("test2"), 100, vec![22,11,12,13]);

        let mut g = Game::create(vec![s, s2], vec![], 0, 11);
        g.advance_turn(vec![(String::from("test1"), Direction::Up), (String::from("test2"), Direction::Up)]);
        dbg!(&g);

        let s = &g.snakes[0];
        let s2 = &g.snakes[1];
        assert_eq!(s.health, 0);
        assert_eq!(s2.health, 99);
    }

    #[test]
    fn head_to_head() {
        let s = Snake::create(String::from("test1"), 100, vec![0,1,2]);
        let s2 = Snake::create(String::from("test2"), 100, vec![12,13,14,15]);

        let mut g = Game::create(vec![s, s2], vec![], 0, 11);
        g.advance_turn(vec![(String::from("test1"), Direction::Up), (String::from("test2"), Direction::Left)]);
        dbg!(&g);

        let s = &g.snakes[0];
        let s2 = &g.snakes[1];
        assert_eq!(s.health, 0);
        assert_eq!(s2.health, 99);
    }

    #[test]
    fn head_to_head_equal_length() {
        let s = Snake::create(String::from("test1"), 100, vec![0,1,2]);
        let s2 = Snake::create(String::from("test2"), 100, vec![12,13,14]);

        let mut g = Game::create(vec![s, s2], vec![], 0, 11);
        g.advance_turn(vec![(String::from("test1"), Direction::Up), (String::from("test2"), Direction::Left)]);
        dbg!(&g);

        g.print();
        let s = &g.snakes[0];
        let s2 = &g.snakes[1];
        
        assert_eq!(s.health, 0);
        assert_eq!(s2.health, 0);
    }

    #[test]
    fn body_collision_on_out_of_bounds_snake() {
        let s = Snake::create(String::from("test1"), 100, vec![0,1,2]);
        let s2 = Snake::create(String::from("test2"), 100, vec![11,12,13]);

        let mut g = Game::create(vec![s, s2], vec![], 0, 11);
        g.advance_turn(vec![(String::from("test1"), Direction::Up), (String::from("test2"), Direction::Left)]);
        dbg!(&g);

        g.print();
        let s = &g.snakes[0];
        let s2 = &g.snakes[1];
        
        assert_eq!(s.health, 0);
        assert_eq!(s2.health, 0);
    }
    // #[test]
    // fn move_back_on_neck() {
    //     let s = Snake::create(String::from("test1"), 100, vec![0,1,2]);

    //     let mut g = Game::create(vec![s], vec![], 0, 11);
    //     g.advance_turn(vec![(String::from("test1"), Direction::Right)]);
    //     dbg!(&g);

    //     g.print();
    //     let s = &g.snakes[0];
        
    //     assert_eq!(s.health, 0);
    // }

}

// TODO: Rename this to 'Board' to be in line with the api
#[derive(Debug, Clone)]
pub struct Game {
    pub occupied: Board,
    pub food: Board,
    pub snakes: Vec<Snake>,
    pub turn: usize,
    pub width: usize,
    // Food spawning settings (optional, following Battlesnake rules)
    pub food_spawn_chance: u8,  // Percentage (0-100), default 15
    pub minimum_food: usize,    // Default 1
}


pub type Move = (String, Direction);

impl Game {
    pub fn create(snakes: Vec<Snake>, food: Vec<u128>, turn: usize, width: usize) -> Self {
        Self::create_with_spawning(
            snakes,
            food,
            turn,
            width,
            DEFAULT_FOOD_SPAWN_CHANCE,
            DEFAULT_MINIMUM_FOOD,
        )
    }

    pub fn create_with_spawning(
        snakes: Vec<Snake>,
        food: Vec<u128>,
        turn: usize,
        width: usize,
        food_spawn_chance: u8,
        minimum_food: usize,
    ) -> Self {
        let occupied = { snakes.iter().fold(0, |a, s| a | s.body_board) };
        let food = { food.iter().fold(0, |a, s| set_index(a, *s, 1)) };

        Self {
            occupied,
            turn,
            food,
            width,
            snakes,
            food_spawn_chance,
            minimum_food,
        }
    }

    /// Counts the number of food items currently on the board.
    pub fn count_food(&self) -> usize {
        self.food.count_ones() as usize
    }

    /// Returns a vector of all unoccupied cell indices.
    /// A cell is unoccupied if it's not part of any snake body and doesn't have food.
    fn get_unoccupied_points(&self) -> Vec<u128> {
        let mut unoccupied = Vec::new();
        let board_size = (self.width * self.width) as u128;

        // Combined bitmap of all occupied cells (snakes + food)
        let occupied_or_food = self.occupied | self.food;

        for pos in 0..board_size {
            if (occupied_or_food >> pos) & 1 == 0 {
                unoccupied.push(pos);
            }
        }

        unoccupied
    }

    /// Returns unoccupied points for food spawning.
    /// Per official Battlesnake rules, this excludes:
    /// - Snake bodies
    /// - Existing food
    /// - Cells adjacent to snake heads (4 cardinal directions)
    fn get_unoccupied_points_for_spawning(&self) -> Vec<u128> {
        let board_size = (self.width * self.width) as u128;
        let width = self.width as u128;

        // Start with occupied cells (snakes + food)
        let mut blocked = self.occupied | self.food;

        // Add cells adjacent to non-eliminated snake heads
        for snake in &self.snakes {
            if snake.is_eliminated() {
                continue;
            }

            let head = snake.head();
            let row = head / width;
            let col = head % width;

            // Block adjacent cells (respecting board boundaries)
            if col > 0 {
                blocked |= 1 << (head - 1); // Left
            }
            if col < width - 1 {
                blocked |= 1 << (head + 1); // Right
            }
            if row > 0 {
                blocked |= 1 << (head - width); // Down
            }
            if row < width - 1 {
                blocked |= 1 << (head + width); // Up
            }
        }

        // Collect unoccupied points
        let mut unoccupied = Vec::new();
        for pos in 0..board_size {
            if (blocked >> pos) & 1 == 0 {
                unoccupied.push(pos);
            }
        }

        unoccupied
    }

    /// Spawns food according to Battlesnake rules.
    /// - If food count < minimum_food, spawn to reach minimum
    /// - Otherwise, food_spawn_chance% probability to spawn 1 food
    ///
    /// Uses the provided RNG for deterministic testing or game simulation.
    pub fn spawn_food<R: Rng>(&mut self, rng: &mut R) {
        let current_food = self.count_food();

        let food_to_spawn = if current_food < self.minimum_food {
            // Spawn enough to reach minimum
            self.minimum_food - current_food
        } else if self.food_spawn_chance > 0 {
            // Random chance to spawn 1 food
            // Official formula: (100 - rand.Intn(100)) < foodSpawnChance
            // Equivalent: roll in 1..=100, spawn if roll < food_spawn_chance
            let roll: u8 = rng.gen_range(1..=100);
            if roll < self.food_spawn_chance { 1 } else { 0 }
        } else {
            0
        };

        if food_to_spawn == 0 {
            return;
        }

        let mut unoccupied = self.get_unoccupied_points_for_spawning();

        if unoccupied.is_empty() {
            return; // No valid spawn locations
        }

        // Shuffle for random placement
        unoccupied.shuffle(rng);

        // Spawn food at the first N unoccupied positions
        for pos in unoccupied.into_iter().take(food_to_spawn) {
            self.food = set_index(self.food, pos, 1);
        }
    }

    /// Advances the game turn with optional food spawning.
    /// This follows the official Battlesnake turn order:
    /// 1. Move snakes
    /// 2. Check collisions/eliminations
    /// 3. Feed snakes
    /// 4. Spawn food (new)
    pub fn advance_turn_with_spawning<R: Rng>(&mut self, moves: Vec<Move>, rng: &mut R) {
        // Execute standard turn logic
        self.advance_turn(moves);

        // Spawn food after all other turn processing
        self.spawn_food(rng);
    }


    pub fn advance_turn(&mut self, moves: Vec<Move>) {
        let mut occupied: u128 = !0;

        let mut eliminated: Vec<String> = vec![];
        for snake in &mut self.snakes {
           if snake.is_eliminated() {
                continue;
           } 

            // Find the move for this snake, or determine a default move
            let s_move = match moves.iter().find(|m| m.0 == snake.id) {
                Some(m) => m.1,
                None => {
                    // If no move specified, continue in the same direction
                    let head = snake.body[0];
                    let neck = snake.body[1];
                    let diff = head as i128 - neck as i128;
                    
                    match diff {
                        d if d == 1 => Direction::Right,
                        d if d == -1 => Direction::Left,
                        d if d == self.width as i128 => Direction::Up,
                        d if d == -(self.width as i128) => Direction::Down,
                        _ => Direction::Up // Fallback if we can't determine direction
                    }
                }
            };

            // check for out of bounds moves
            if !is_dir_valid(&snake, &s_move) {
                eliminated.push(snake.id.clone());
                snake.move_tail();
                continue;
            }

            snake.move_tail();
            snake.move_head(&s_move, self.width as u128);

            snake.health -= 1;

            occupied = occupied ^ snake.body_board ^ snake.head_board;
        }

        let mut eaten_food: Vec<u128>= vec![];
        for snake in &mut self.snakes {
            if snake.head_board & self.food > 0 {
                // println!("feeding {}", snake.id);
                snake.feed();
                eaten_food.push(snake.head());
            }
        }

        for eaten in eaten_food {
            self.food = set_index(self.food, eaten, 0);
        }

        // Biggest snake should win
        let mut sorted_snakes: Vec<_> = self.snakes.iter()
            .filter(|s| !s.is_eliminated())
            .map(|s| (s.id.clone(), s.head_board, s.length()))
            .collect();

        sorted_snakes.sort_unstable_by(|a,b| a.2.cmp(&b.2));


        // collisions
        for snake in &mut self.snakes {
            if snake.is_eliminated() {
                continue;
            }

            let mut had_h2h = false;

            for snake_2 in sorted_snakes.as_slice() {
                if snake_2.0 == snake.id {
                    continue;
                }

                if snake.head_board & snake_2.1 > 0 {
                    had_h2h = true;
                    if snake_2.2 >= snake.length() {
                        // println!("{} head to head!", snake.id);
                        eliminated.push(snake.id.clone());
                    }
                }
            }

            // If the snake had a head-to-head, then we know the check against
            // `occupied` will return greater than 0, so we skip the check.
            if !had_h2h && snake.head_board & occupied > 0 {
                // println!("{:b}", occupied);
                // println!("{} collision!", snake.id);
                eliminated.push(snake.id.clone());
            }
        }

        // eliminations from collisions
        for elim in eliminated{
           for snake in &mut self.snakes {
                if snake.id == elim {
                    snake.health = 0;
                }
            }
        }

        // Recompute occupied: bit=1 means snake body is present
        let mut new_occupied: u128 = 0;
        for snake in &self.snakes {
            if !snake.is_eliminated() {
                new_occupied = new_occupied | snake.head_board | snake.body_board;
            }
        }

        self.occupied = new_occupied;
        self.turn += 1;
    }

    pub fn print(&self) {
        let mut game: Vec<String> = vec![];
        for y in (0..WIDTH).rev() {
            for x in 0..WIDTH {
                let cell = y*WIDTH + x;
                game.push(format!(" {} ", self.cell_to_string(cell)));
                // game.push(format!("{}",  if cell % 11 == 0 { "\n" } else { "" }));
            }
            game.push("\n".to_string());
        }

        print!("{}", game.iter().fold(String::new(), |a,c| a + c));
    }

    fn cell_to_string(&self, index: u128) -> String {
        let b = set_index(0, index, 1);

        for (i, s) in self.snakes.iter().enumerate() {
            if !s.is_eliminated() && b & s.head_board > 0 {
                return format!("{}h", i);
            }
        }

        for (i, s) in self.snakes.iter().enumerate() {
            if !s.is_eliminated() && b & s.body_board > 0 {
                return format!("{}s", i);
            }
        }

        if b & self.food > 0 {
            return format!("@@");
        }
        
        return String::from("__");
    }

    pub fn calculate_voronoi_scores(&self) -> std::collections::HashMap<String, i32> {
        use std::collections::{HashMap, VecDeque};

        let mut scores = vec![0i32; self.snakes.len()]; // Use Vec instead of HashMap
        let mut distances: HashMap<u128, u32> = HashMap::new();
        let mut owners: HashMap<u128, usize> = HashMap::new(); // Store snake index instead of ID
        let mut queue: VecDeque<(u128, usize, u32)> = VecDeque::new(); // (pos, snake_idx, distance)

        // Initialize queue with all snake heads at distance 0
        for (snake_idx, snake) in self.snakes.iter().enumerate() {
            if !snake.is_eliminated() {
                let head = snake.head();
                queue.push_back((head, snake_idx, 0));
                distances.insert(head, 0);
                owners.insert(head, snake_idx);
            }
        }

        // BFS from all snake heads simultaneously
        let width = self.width as u128; // Hoist constant outside loop

        while let Some((pos, snake_idx, dist)) = queue.pop_front() {
            let pos_board = 1u128 << pos; // Convert position to bitboard for boundary checks
            let next_dist = dist + 1;

            // Unroll direction loop for better performance
            // Check Up
            if pos_board & TOP_MASK == 0 {
                let next_pos = pos + width;
                if self.occupied & (1 << next_pos) == 0 {
                    match distances.get(&next_pos) {
                        None => {
                            distances.insert(next_pos, next_dist);
                            owners.insert(next_pos, snake_idx);
                            queue.push_back((next_pos, snake_idx, next_dist));
                        }
                        Some(&existing_dist) if next_dist == existing_dist => {
                            if let Some(&existing_owner) = owners.get(&next_pos) {
                                if existing_owner != snake_idx {
                                    owners.remove(&next_pos);
                                }
                            }
                        }
                        _ => {} // Shorter path exists, skip
                    }
                }
            }

            // Check Down
            if pos_board & BOTTOM_MASK == 0 {
                let next_pos = pos - width;
                if self.occupied & (1 << next_pos) == 0 {
                    match distances.get(&next_pos) {
                        None => {
                            distances.insert(next_pos, next_dist);
                            owners.insert(next_pos, snake_idx);
                            queue.push_back((next_pos, snake_idx, next_dist));
                        }
                        Some(&existing_dist) if next_dist == existing_dist => {
                            if let Some(&existing_owner) = owners.get(&next_pos) {
                                if existing_owner != snake_idx {
                                    owners.remove(&next_pos);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Check Left
            if pos_board & LEFT_MASK == 0 {
                let next_pos = pos - 1;
                if self.occupied & (1 << next_pos) == 0 {
                    match distances.get(&next_pos) {
                        None => {
                            distances.insert(next_pos, next_dist);
                            owners.insert(next_pos, snake_idx);
                            queue.push_back((next_pos, snake_idx, next_dist));
                        }
                        Some(&existing_dist) if next_dist == existing_dist => {
                            if let Some(&existing_owner) = owners.get(&next_pos) {
                                if existing_owner != snake_idx {
                                    owners.remove(&next_pos);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Check Right
            if pos_board & RIGHT_MASK == 0 {
                let next_pos = pos + 1;
                if self.occupied & (1 << next_pos) == 0 {
                    match distances.get(&next_pos) {
                        None => {
                            distances.insert(next_pos, next_dist);
                            owners.insert(next_pos, snake_idx);
                            queue.push_back((next_pos, snake_idx, next_dist));
                        }
                        Some(&existing_dist) if next_dist == existing_dist => {
                            if let Some(&existing_owner) = owners.get(&next_pos) {
                                if existing_owner != snake_idx {
                                    owners.remove(&next_pos);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Count cells owned by each snake
        for &snake_idx in owners.values() {
            scores[snake_idx] += 1;
        }

        // Convert Vec scores to HashMap with snake IDs
        let mut result: HashMap<String, i32> = HashMap::new();
        for (idx, &score) in scores.iter().enumerate() {
            if score > 0 {
                result.insert(self.snakes[idx].id.clone(), score);
            }
        }

        result
    }
}

pub fn is_dir_valid(snake: &Snake, dir: &Direction) -> bool {
    let mask = match dir {
        Direction::Left => LEFT_MASK,
        Direction::Right => RIGHT_MASK,
        Direction::Up => TOP_MASK,
        Direction::Down => BOTTOM_MASK,
    };

    snake.head_board & mask == 0
}

pub fn set_index(board: Board, pos: u128, to: u128) -> u128 {
    let mask = 1 << pos;
    (board & !mask) | (to << pos)
}

pub fn dir_to_index(head: u128, dir: &Direction, width: u128) -> u128 {
   match dir {
            Direction::Left => head - 1,
            Direction::Right => head + 1,
            Direction::Up => head + width,
            Direction::Down => head - width,
    }
}

// Default food spawning settings (per official Battlesnake rules)
pub const DEFAULT_FOOD_SPAWN_CHANCE: u8 = 15;
pub const DEFAULT_MINIMUM_FOOD: usize = 1;

#[cfg(test)]
mod food_spawning_tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn count_food_empty_board() {
        let g = Game::create(vec![], vec![], 0, 11);
        assert_eq!(g.count_food(), 0);
    }

    #[test]
    fn count_food_multiple_items() {
        let g = Game::create(vec![], vec![0, 5, 10, 60], 0, 11);
        assert_eq!(g.count_food(), 4);
    }

    #[test]
    fn get_unoccupied_excludes_snake_bodies() {
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let unoccupied = g.get_unoccupied_points();

        // Snake body positions should be excluded
        assert!(!unoccupied.contains(&0));
        assert!(!unoccupied.contains(&1));
        assert!(!unoccupied.contains(&2));
        // Other positions should be included
        assert!(unoccupied.contains(&3));
        assert!(unoccupied.contains(&60));
    }

    #[test]
    fn get_unoccupied_excludes_existing_food() {
        let g = Game::create(vec![], vec![5, 10, 60], 0, 11);
        let unoccupied = g.get_unoccupied_points();

        // Food positions should be excluded
        assert!(!unoccupied.contains(&5));
        assert!(!unoccupied.contains(&10));
        assert!(!unoccupied.contains(&60));
        // Other positions should be included
        assert!(unoccupied.contains(&0));
        assert!(unoccupied.contains(&3));
    }

    #[test]
    fn get_unoccupied_excludes_both_snakes_and_food() {
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g = Game::create(vec![s], vec![5, 10], 0, 11);
        let unoccupied = g.get_unoccupied_points();

        // Snake and food positions should be excluded
        assert!(!unoccupied.contains(&0));
        assert!(!unoccupied.contains(&1));
        assert!(!unoccupied.contains(&2));
        assert!(!unoccupied.contains(&5));
        assert!(!unoccupied.contains(&10));
        // Other positions should be included
        assert!(unoccupied.contains(&3));
        assert!(unoccupied.contains(&60));
        // Total should be 121 - 3 (snake) - 2 (food) = 116
        assert_eq!(unoccupied.len(), 116);
    }

    #[test]
    fn spawn_food_reaches_minimum_when_below() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = Game::create_with_spawning(vec![], vec![], 0, 11, 0, 3);

        assert_eq!(g.count_food(), 0);
        g.spawn_food(&mut rng);
        assert_eq!(g.count_food(), 3);
    }

    #[test]
    fn spawn_food_no_spawn_at_zero_chance() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = Game::create_with_spawning(vec![], vec![0], 0, 11, 0, 1);

        assert_eq!(g.count_food(), 1);
        g.spawn_food(&mut rng);
        // With 0% chance and already at minimum, no spawn
        assert_eq!(g.count_food(), 1);
    }

    #[test]
    fn spawn_food_always_spawns_at_100_percent() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut g = Game::create_with_spawning(vec![], vec![0], 0, 11, 100, 1);

        assert_eq!(g.count_food(), 1);
        g.spawn_food(&mut rng);
        // With 100% chance, should always spawn 1 more
        assert_eq!(g.count_food(), 2);
    }

    #[test]
    fn spawn_food_deterministic_with_seed() {
        // Same seed should produce same results
        let mut rng1 = StdRng::seed_from_u64(12345);
        let mut rng2 = StdRng::seed_from_u64(12345);

        let mut g1 = Game::create_with_spawning(vec![], vec![], 0, 11, 100, 2);
        let mut g2 = Game::create_with_spawning(vec![], vec![], 0, 11, 100, 2);

        g1.spawn_food(&mut rng1);
        g2.spawn_food(&mut rng2);

        // Both should have food at the same positions
        assert_eq!(g1.food, g2.food);
    }

    #[test]
    fn spawn_food_no_spawn_on_full_board() {
        let mut rng = StdRng::seed_from_u64(42);
        // Create a snake that fills the entire 11x11 board (121 cells)
        let body: Vec<u128> = (0..121).collect();
        let s = Snake::create(String::from("test"), 100, body);
        let mut g = Game::create_with_spawning(vec![s], vec![], 0, 11, 100, 5);

        g.spawn_food(&mut rng);
        // No valid spawn locations, so no food spawned
        assert_eq!(g.count_food(), 0);
    }

    #[test]
    fn spawn_food_spawns_at_unoccupied_positions() {
        let mut rng = StdRng::seed_from_u64(42);
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let mut g = Game::create_with_spawning(vec![s], vec![], 0, 11, 100, 2);

        g.spawn_food(&mut rng);

        // Food should not overlap with snake
        assert_eq!(g.food & g.occupied, 0);
        assert_eq!(g.count_food(), 2);
    }

    #[test]
    fn spawn_food_probability_works_over_many_trials() {
        // With 50% chance, roughly half should spawn over many trials
        let mut spawn_count = 0;
        let trials = 1000;

        for seed in 0..trials {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut g = Game::create_with_spawning(vec![], vec![0], 0, 11, 50, 1);
            let initial_food = g.count_food();
            g.spawn_food(&mut rng);
            if g.count_food() > initial_food {
                spawn_count += 1;
            }
        }

        // Should be roughly 500 (+/- reasonable variance)
        // Allow for 40-60% range to avoid flaky tests
        assert!(spawn_count > 400 && spawn_count < 600,
            "Expected ~500 spawns with 50% chance, got {}", spawn_count);
    }

    #[test]
    fn advance_turn_with_spawning_integrates_correctly() {
        let mut rng = StdRng::seed_from_u64(42);
        let s = Snake::create(String::from("test"), 100, vec![5, 4, 3]);
        let mut g = Game::create_with_spawning(vec![s], vec![], 0, 11, 100, 1);

        g.advance_turn_with_spawning(
            vec![(String::from("test"), Direction::Up)],
            &mut rng
        );

        // Turn should advance
        assert_eq!(g.turn, 1);
        // Snake should have moved
        assert_eq!(g.snakes[0].body[0], 16); // 5 + 11 = 16
        // Food should have spawned (100% chance, minimum 1)
        assert!(g.count_food() >= 1);
    }

    #[test]
    fn get_unoccupied_for_spawning_excludes_head_adjacent() {
        // Snake head at position 60 (center-ish of 11x11 board)
        let s = Snake::create(String::from("test"), 100, vec![60, 59, 58]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let unoccupied = g.get_unoccupied_points_for_spawning();

        // Snake body positions should be excluded
        assert!(!unoccupied.contains(&60)); // head
        assert!(!unoccupied.contains(&59)); // body
        assert!(!unoccupied.contains(&58)); // tail

        // Head-adjacent positions should be excluded
        assert!(!unoccupied.contains(&61)); // right of head
        assert!(!unoccupied.contains(&71)); // up from head (60 + 11)
        assert!(!unoccupied.contains(&49)); // down from head (60 - 11)
        // Note: 59 (left of head) is already body

        // Non-adjacent positions should be included
        assert!(unoccupied.contains(&0));
        assert!(unoccupied.contains(&120));
    }

    #[test]
    fn get_unoccupied_for_spawning_respects_board_edges() {
        // Snake head at corner position 0
        let s = Snake::create(String::from("test"), 100, vec![0, 1, 2]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let unoccupied = g.get_unoccupied_points_for_spawning();

        // Head-adjacent within bounds should be excluded
        assert!(!unoccupied.contains(&11)); // up from head
        // Note: 1 (right of head) is already body
        // Left and down would be out of bounds, so no crash

        // Far positions should be included
        assert!(unoccupied.contains(&120));
        assert!(unoccupied.contains(&60));
    }

    #[test]
    fn get_unoccupied_for_spawning_ignores_eliminated_snakes() {
        let mut s = Snake::create(String::from("test"), 100, vec![60, 59, 58]);
        s.health = 0; // Eliminate the snake
        let g = Game::create(vec![s], vec![], 0, 11);

        // Note: occupied is computed at create time, before elimination
        // But get_unoccupied_points_for_spawning checks is_eliminated()
        let unoccupied = g.get_unoccupied_points_for_spawning();

        // Head-adjacent should NOT be excluded for eliminated snakes
        assert!(unoccupied.contains(&61)); // right of head
        assert!(unoccupied.contains(&71)); // up from head
        assert!(unoccupied.contains(&49)); // down from head
    }

    #[test]
    fn spawn_food_avoids_head_adjacent_cells() {
        let mut rng = StdRng::seed_from_u64(42);
        // Snake in center of small area to block many spawn points
        let s = Snake::create(String::from("test"), 100, vec![60, 59, 58]);
        let mut g = Game::create_with_spawning(vec![s], vec![], 0, 11, 100, 10);

        g.spawn_food(&mut rng);

        // Food should not be adjacent to head
        let head = 60u128;
        let adjacent_mask = (1 << (head + 1)) | (1 << (head + 11)) | (1 << (head - 11));
        // Note: head - 1 = 59 is body, already excluded

        assert_eq!(g.food & adjacent_mask, 0,
            "Food should not spawn adjacent to snake head");
    }
}

#[cfg(test)]
mod voronoi_tests {
    use super::*;

    #[test]
    fn voronoi_single_snake_empty_board() {
        let s = Snake::create(String::from("test"), 100, vec![60, 59, 58]);
        let g = Game::create(vec![s], vec![], 0, 11);
        let scores = g.calculate_voronoi_scores();

        // Snake head counts + all unoccupied cells (1 head + 118 unoccupied = 119)
        // Body cells (59, 58) are in occupied and not counted separately
        assert_eq!(scores.get("test").copied().unwrap_or(0), 119);
    }

    #[test]
    fn voronoi_two_snakes_opposite_corners() {
        // Snake 1 at bottom-left corner
        let s1 = Snake::create(String::from("snake1"), 100, vec![0, 1, 2]);
        // Snake 2 at top-right corner
        let s2 = Snake::create(String::from("snake2"), 100, vec![120, 119, 118]);
        let g = Game::create(vec![s1, s2], vec![], 0, 11);
        let scores = g.calculate_voronoi_scores();

        let s1_score = scores.get("snake1").copied().unwrap_or(0);
        let s2_score = scores.get("snake2").copied().unwrap_or(0);

        // Both should have territory
        assert!(s1_score > 0);
        assert!(s2_score > 0);

        // Equidistant cells should NOT be counted for either snake
        // Total available: 2 heads + 115 unoccupied cells = 117
        // Since there are equidistant cells, sum should be LESS than 117
        assert!(s1_score + s2_score < 117);

        // They should have roughly equal territory (symmetric positions)
        // Allow for some asymmetry due to board geometry
        let diff = (s1_score - s2_score).abs();
        assert!(diff <= 2, "Territory difference too large: s1={}, s2={}, diff={}", s1_score, s2_score, diff);
    }

    #[test]
    fn voronoi_sum_excludes_equidistant() {
        let s1 = Snake::create(String::from("s1"), 100, vec![60, 61, 62]);
        let s2 = Snake::create(String::from("s2"), 100, vec![10, 11, 12]);
        let g = Game::create(vec![s1, s2], vec![], 0, 11);
        let scores = g.calculate_voronoi_scores();

        let total: i32 = scores.values().sum();
        // Max possible: 2 heads + 115 unoccupied cells = 117
        // But equidistant cells are excluded, so total < 117
        assert!(total < 117, "Total should be less than 117 due to equidistant cells");
        assert!(total > 0, "Both snakes should control some territory");
    }

    #[test]
    fn voronoi_eliminated_snakes_dont_control() {
        let mut s1 = Snake::create(String::from("s1"), 100, vec![60, 61, 62]);
        s1.health = 100;
        let s2 = Snake::create(String::from("s2"), 0, vec![10, 11, 12]); // eliminated

        let g = Game::create(vec![s1, s2], vec![], 0, 11);
        let scores = g.calculate_voronoi_scores();

        // Only s1 should have territory
        assert!(scores.get("s1").copied().unwrap_or(0) > 0);
        assert_eq!(scores.get("s2").copied().unwrap_or(0), 0);
    }

    #[test]
    fn voronoi_snake_body_blocks_expansion() {
        // Create a long snake that acts as a wall
        let wall = Snake::create(String::from("wall"), 100,
            vec![55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65]); // horizontal line
        let s1 = Snake::create(String::from("s1"), 100, vec![5, 6, 7]);
        let s2 = Snake::create(String::from("s2"), 100, vec![115, 116, 117]);

        let g = Game::create(vec![wall, s1, s2], vec![], 0, 11);
        let scores = g.calculate_voronoi_scores();

        // Wall should divide the board
        let s1_score = scores.get("s1").copied().unwrap_or(0);
        let s2_score = scores.get("s2").copied().unwrap_or(0);
        let wall_score = scores.get("wall").copied().unwrap_or(0);

        // All snakes should control some territory
        assert!(s1_score > 0);
        assert!(s2_score > 0);
        assert!(wall_score > 0);

        // Max possible: 3 heads + 104 unoccupied cells = 107
        // But equidistant cells are excluded, so total <= 107
        let total = s1_score + s2_score + wall_score;
        assert!(total <= 107);
    }

    #[test]
    fn voronoi_equidistant_cells_not_counted() {
        // Two snakes side by side - many cells will be equidistant
        let s1 = Snake::create(String::from("s1"), 100, vec![50, 49, 48]);
        let s2 = Snake::create(String::from("s2"), 100, vec![52, 53, 54]);
        let g = Game::create(vec![s1, s2], vec![], 0, 11);
        let scores = g.calculate_voronoi_scores();

        let s1_score = scores.get("s1").copied().unwrap_or(0);
        let s2_score = scores.get("s2").copied().unwrap_or(0);
        let total = s1_score + s2_score;

        // Total available: 2 heads + (121 - 6) unoccupied = 117
        // But cells equidistant from both heads should NOT be counted
        assert!(total < 117, "Equidistant cells should not be counted. Got total={}", total);

        // Both should still have some territory
        assert!(s1_score > 0);
        assert!(s2_score > 0);
    }

    #[test]
    fn voronoi_empty_game() {
        let g = Game::create(vec![], vec![], 0, 11);
        let scores = g.calculate_voronoi_scores();

        // Empty game should return empty HashMap
        assert_eq!(scores.len(), 0);
    }
}
