pub mod battlesnake;
pub mod bitboard;
pub mod rhea;

use crate::battlesnake::{MoveRequest, MoveResponse, SnakeInfo};
use crate::bitboard::{Direction, Game, Move, Snake};
use crate::rhea::RHEA;
use actix_web::{
    get, middleware, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder,
};

async fn info() -> impl Responder {
    HttpResponse::Ok().json(SnakeInfo {
        color: "blue".to_string(),
        head: "beluga".to_string(),
        tail: "curled".to_string(),
    })
}

async fn get_move(state: web::Json<MoveRequest>) -> impl Responder {
    println!("{:?}", state);

    HttpResponse::Ok().json(MoveResponse {
        r#move: "left".to_string(),
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
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    HttpServer::new(|| {
        App::new()
            .wrap(middleware::Logger::default())
            .route("/move", web::post().to(get_move))
            .route("/", web::get().to(info))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
