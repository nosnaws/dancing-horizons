type Board = u128;

const WIDTH: u128 = 11;

const LEFT_MASK: u128 = 
0b0000000_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001;

const RIGHT_MASK: u128 = 
0b0000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000;

const TOP_MASK: u128 = 
0b0000000_11111111111_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000;

const BOTTOM_MASK: u128 = 
0b0000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_00000000000_11111111111;

#[derive(Debug)]
pub struct Snake {
    pub id: String,
    pub health: u128,
    pub body: Vec<u128>,
    pub head_board: Board,
    pub body_board: Board,
    pub length: usize,
}

pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Snake {
    pub fn create(id: String, health: u128, body: Vec<u128>) -> Self {
        let head_board = set_index(0, body[0], 1);
        // Not sure I want the clone here, need to read up on that
        let body_board = body
            .clone()
            .into_iter()
            .fold(0, |acc, p| set_index(acc, p, 1));

        Self {
            id,
            health,
            body,
            head_board,
            body_board,
            length: body.len(),
        }
    }

    pub fn length(&self) -> usize {
        self.length
    }

    fn head(&self) -> u128 {
        self.body[0]
    }

    pub fn move_in_dir(&mut self, dir: &Direction, width: u128) {
        let tail = self.body.pop().unwrap();
        self.body_board = set_index(self.body_board, tail, 0);


        // check for out of bounds moves
        if !is_dir_valid(self, dir) {
            self.health = 0;
            return;
        }

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
    use crate::Snake;

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
    fn move_up() {
        let mut s = Snake::create(String::from("test"), 100, vec![0,1,2]);

        s.move_in_dir(&super::Direction::Up, 11);

        assert_eq!(s.body, vec![11,0,1]);
        // assert_eq!(s.head_board, 2048);
        assert_eq!(s.head_board, 0b100000000000);
        assert_eq!(s.body_board, 0b100000000011);
    }
}

pub struct Game {
    pub empty: Board,
    pub food: Board,
    pub snakes: Vec<Snake>,
    pub turn: usize,
    pub width: usize,
}


type Move = (String, Direction);

impl Game {
    pub fn create(snakes: Vec<Snake>, food: Vec<u128>, turn: usize, width: usize) -> Self {
        let empty = { snakes.iter().fold(0, |a, s| a | s.body_board) };
        let food = { food.iter().fold(0, |a, s| a | s) };

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

        for snake in &mut self.snakes {
           if snake.health < 1 {
                continue;
           } 

            let s_move = moves.iter().find(|m| m.0 == snake.id).unwrap();
             
            snake.move_in_dir(&s_move.1, WIDTH);

            snake.health -= 1;

            empty = empty ^ snake.body_board ^ snake.head_board;
        }

        let mut new_empty: u128 = !0;
        // kill conditions
        let snake_heads = self.snakes.iter().map(|s| (s.id, s.head_board));
        for snake in &mut self.snakes {
            if snake.health < 1 {
                continue;
            }

            if snake.head_board & empty == 0 {
                snake.health == 0;
            }

            let other_snakes = self.snakes.iter().filter(|s| s.id != snake.id);
            
            // AAAAAAAAAAAAAAAAGGGGGGGGGGGHHHHHHHHHHHHHH
            for snake_2 in &self.snakes {
                if snake_2.health < 1 {
                    continue;
                }
                if snake_2.id == snake.id {
                    continue;
                }

                if snake.head_board & snake_2.head_board > 0 {
                    if snake_2.length() >= snake.length() {
                        snake.health = 0; 
                    }
                }
            }

            if snake.head_board & self.food > 0 {
                snake.health = 100;
                self.food = set_index(self.food, snake.head(), 0);
            }

            if snake.health > 0 {
                new_empty = new_empty ^ snake.head_board ^ snake.body_board;
            }
        }

        self.empty = new_empty;
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

