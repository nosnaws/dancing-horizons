use crate::bitboard;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SnakeInfo {
    pub color: String,
    pub head: String,
    pub tail: String,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct MoveResponse {
    pub r#move: String,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct MoveRequest {
    pub game: Game,
    pub turn: i32,
    pub board: Board,
    pub you: Battlesnake,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct Ruleset {
    pub name: String,
    pub version: String,
    pub settings: RulesetSettings,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct RulesetSettings {
    pub food_spawn_chance: u32,
    pub minimum_food: u32,
    pub hazard_damage_per_turn: u32,
    pub hazard_map: String,
    pub hazard_map_author: String,
    pub royale: RoyaleSettings,
    pub squad: SquadSettings,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct SquadSettings {
    pub allow_body_collisions: bool,
    pub shared_elimination: bool,
    pub shared_health: bool,
    pub shared_length: bool,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct RoyaleSettings {
    pub shrink_every_n_turns: u32,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct Game {
    pub id: String,
    // pub ruleset: Ruleset,
    pub map: String,
    pub timeout: i32,
    pub source: String,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct Battlesnake {
    pub id: String,
    pub name: String,
    pub health: i32,
    pub body: Vec<Coordinate>,
    pub latency: String,
    pub head: Coordinate,
    pub length: usize,
    pub shout: String,
    pub squad: String,
    pub customizations: SnakeInfo,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct Coordinate {
    pub x: i32,
    pub y: i32,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub struct Board {
    pub height: i32,
    pub width: i32,
    pub food: Vec<Coordinate>,
    pub hazards: Vec<Coordinate>,
    pub snakes: Vec<Battlesnake>,
}

pub fn direction_to_string(d: bitboard::Direction) -> String {
    match d {
        bitboard::Direction::Up => "up".to_string(),
        bitboard::Direction::Down => "down".to_string(),
        bitboard::Direction::Right => "right".to_string(),
        bitboard::Direction::Left => "left".to_string(),
    }
}

pub fn request_to_game(req: &MoveRequest) -> bitboard::Game {
    let food = req
        .board
        .food
        .iter()
        .map(|c| coord_to_index(c, req.board.width))
        .collect();

    let snakes = req
        .board
        .snakes
        .iter()
        .map(|bs| battlesnake_to_snake(bs, req.board.width))
        .collect();

    bitboard::Game::create(
        snakes,
        food,
        req.turn.try_into().unwrap(),
        req.board.width.try_into().unwrap(),
    )
}

pub fn battlesnake_to_snake(bs: &Battlesnake, width: i32) -> bitboard::Snake {
    let body = bs.body.iter().map(|c| coord_to_index(c, width)).collect();

    bitboard::Snake::create(bs.id.clone(), bs.health, body)
}

fn coord_to_index(c: &Coordinate, width: i32) -> u128 {
    (c.y * width + c.x).try_into().unwrap()
}
