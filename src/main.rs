pub mod battlesnake;
pub mod bitboard;
pub mod rhea;

use crate::battlesnake::{MoveRequest, MoveResponse, SnakeInfo};
use crate::bitboard::{Direction, Game, Move, Snake};
use crate::rhea::RHEA;
use actix_web::{
    get, middleware, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder,
};
use std::collections::HashMap;
use std::sync::Mutex;

struct DHState {
    games: Mutex<HashMap<String, RHEA>>,
}

async fn info() -> impl Responder {
    HttpResponse::Ok().json(SnakeInfo {
        color: "#73937E".to_string(), // cambridge blue
        head: "space-helmet".to_string(),
        tail: "mlh-gene".to_string(),
    })
}

async fn get_move(context: web::Data<DHState>, state: web::Json<MoveRequest>) -> impl Responder {
    println!("Turn: {:?}", state.turn);

    let game = battlesnake::request_to_game(&state.0);
    let mut all_games = context.games.lock().expect("Failed to get games lock?");

    let mut ga: RHEA = match all_games.get(&state.game.id) {
        Some(g) => g.clone().update_game(game),
        None => RHEA::create(game, state.you.id.clone()),
    };

    // let mut ga = RHEA::create(game, state.you.id.clone());
    for _ in 0..500 {
        ga = ga.evolve();
    }

    let best_move = ga.get_move();

    all_games.insert(state.game.id.clone(), ga);
    println!("Response: {:?}", best_move);

    HttpResponse::Ok().json(MoveResponse {
        r#move: battlesnake::direction_to_string(best_move),
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // let s = Snake::create(String::from("test1"), 100, vec![0, 1, 2]);
    // // let s2 = Snake::create(String::from("test2"), 100, vec![10, 9, 8]);

    // let g = Game::create(vec![s], vec![], 0, 11);
    // g.print();

    // let mut r = RHEA::create(g, String::from("test1"));

    // for _i in 0..555 {
    //     r = r.evolve();
    // }
    // println!("{:?}", r);
    // let best_move = r.get_move();

    // println!("move: {:?}", best_move);
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));

    let context = web::Data::new(DHState {
        games: Mutex::new(HashMap::new()),
    });

    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .app_data(context.clone())
            .route("/move", web::post().to(get_move))
            .route("/", web::get().to(info))
    })
    .bind(("127.0.0.1", 9999))?
    .run()
    .await
}
