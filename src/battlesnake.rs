use serde::{Deserialize, Serialize};

#[derive(Serialize, Debug)]
pub struct SnakeInfo {
    pub color: String,
    pub head: String,
    pub tail: String,
}

#[derive(Serialize, Debug)]
pub struct MoveResponse {
    pub r#move: String,
}

#[derive(Deserialize, Debug)]
pub struct MoveRequest {
    pub game: Game,
    pub turn: i32,
    pub board: Board,
    pub you: Battlesnake,
}

#[derive(Deserialize, Debug)]
pub struct Ruleset {
    pub name: String,
    pub version: String,
}

#[derive(Deserialize, Debug)]
pub struct Game {
    pub id: String,
    pub ruleset: Ruleset,
    pub map: String,
    pub timeout: i32,
    pub source: String,
}

#[derive(Deserialize, Debug)]
pub struct Battlesnake {
    pub id: String,
    pub name: String,
    pub health: i32,
    pub body: Vec<Coordinate>,
    pub latency: i32,
    pub head: Coordinate,
    pub length: usize,
}

#[derive(Deserialize, Debug)]
pub struct Coordinate {
    pub x: i32,
    pub y: i32,
}

#[derive(Deserialize, Debug)]
pub struct Board {
    pub height: i32,
    pub width: i32,
    pub food: Vec<Coordinate>,
    pub hazards: Vec<Coordinate>,
    pub snakes: Vec<Battlesnake>,
}
