use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
// BOARD
// 111 | 112 | 113 | 114 | 115 | 116 | 117 | 118 | 119 | 120 | 121
// 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 | 110
// 089 | 090 | 091 | 092 | 093 | 094 | 095 | 096 | 097 | 098 | 099
// 078 | 079 | 080 | 081 | 082 | 083 | 084 | 085 | 086 | 087 | 088
// 067 | 068 | 069 | 070 | 071 | 072 | 073 | 074 | 075 | 076 | 077
// 055 | 056 | 057 | 058 | 059 | 061 | 062 | 063 | 064 | 065 | 066
// 044 | 045 | 046 | 047 | 048 | 049 | 050 | 051 | 052 | 053 | 054
// 033 | 034 | 035 | 036 | 037 | 038 | 039 | 040 | 041 | 042 | 043
// 022 | 023 | 024 | 025 | 026 | 027 | 028 | 029 | 030 | 031 | 032
// 011 | 012 | 013 | 014 | 015 | 016 | 017 | 018 | 019 | 020 | 021
// 000 | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010
type Board = u128;

const WIDTH: u128 = 11;
const MAX_HEALTH: i32= 100;

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
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
        assert_eq!(g.empty, !0);
    }

    #[test]
    fn out_of_bounds_bottom() {
        let s = Snake::create(String::from("test"), 100, vec![0,1,2]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Down)]);
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
        assert_eq!(g.empty, !0);
    }

    #[test]
    fn out_of_bounds_right() {
        let s = Snake::create(String::from("test"), 100, vec![10,9,8]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Right)]);
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
        assert_eq!(g.empty, !0);
    }

    #[test]
    fn out_of_bounds_top() {
        let s = Snake::create(String::from("test"), 100, vec![111,112,113]);

        let mut g = Game::create(vec![s], vec![11], 0, 11);
        g.advance_turn(vec![(String::from("test"), Direction::Up)]);
        dbg!(&g);

        let s = &g.snakes[0];
        assert_eq!(s.health, 0);
        assert_eq!(g.empty, !0);
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
    pub empty: Board,
    pub food: Board,
    pub snakes: Vec<Snake>,
    pub turn: usize,
    pub width: usize,
}


pub type Move = (String, Direction);

impl Game {
    pub fn create(snakes: Vec<Snake>, food: Vec<u128>, turn: usize, width: usize) -> Self {
        let empty = { snakes.iter().fold(0, |a, s| a | s.body_board) };
        let food = { food.iter().fold(0, |a, s| set_index(a, *s, 1)) };

        Self {
            empty,
            turn,
            food,
            width,
            snakes,
        }
    }


    pub fn advance_turn(&mut self, moves: Vec<Move>) {
        let mut empty: u128 = !0;

        let mut eliminated: Vec<String> = vec![];
        for snake in &mut self.snakes {
           if snake.is_eliminated() {
                continue;
           } 

            let s_move = moves.iter().find(|m| m.0 == snake.id).unwrap();
            // check for out of bounds moves
            if !is_dir_valid(&snake, &s_move.1) {
                // println!("{} out of bounds!", snake.id);
                eliminated.push(snake.id.clone());
                // snake.health = 0;
                snake.move_tail();
                continue;
            }

             
            snake.move_tail();
            snake.move_head(&s_move.1, WIDTH);

            snake.health -= 1;

            empty = empty ^ snake.body_board ^ snake.head_board;
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
            // `empty` will return greater than 0, so we skip the check.
            if !had_h2h && snake.head_board & empty > 0 {
                // println!("{:b}", empty);
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

        let mut new_empty: u128 = !0;
        for snake in &self.snakes {
            if !snake.is_eliminated() {
                new_empty = new_empty ^ snake.head_board ^ snake.body_board;
            }
        }

        self.empty = new_empty;
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
}

fn is_dir_valid(snake: &Snake, dir: &Direction) -> bool {
    let mask = match dir {
        Direction::Left => LEFT_MASK,
        Direction::Right => RIGHT_MASK,
        Direction::Up => TOP_MASK,
        Direction::Down => BOTTOM_MASK,
    };

    snake.head_board & mask == 0
}

fn set_index(board: Board, pos: u128, to: u128) -> u128 {
    let mask = 1 << pos;
    (board & !mask) | (to << pos)
}

fn dir_to_index(head: u128, dir: &Direction, width: u128) -> u128 {
   match dir {
            Direction::Left => head - 1,
            Direction::Right => head + 1,
            Direction::Up => head + width,
            Direction::Down => head - width,
    }
}

