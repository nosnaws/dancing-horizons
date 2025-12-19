pub mod battlesnake;
pub mod bitboard;
pub mod rhea;

use crate::battlesnake::{MoveRequest, MoveResponse, SnakeInfo};
use crate::rhea::{Algorithm, AlgorithmInstance, OpponentModel, RheaConfig};
use actix_web::{
    middleware, web, App, HttpResponse, HttpServer, Responder,
};
use clap::Parser;
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Parser, Debug)]
#[command(name = "dancing-horizons")]
#[command(about = "Battlesnake AI server using RHEA or Negamax algorithms")]
struct Args {
    /// Algorithm: "territorial" or "negamax"
    #[arg(short, long, default_value = "territorial")]
    algorithm: String,

    /// Port to bind the server to
    #[arg(short, long, default_value_t = 8080)]
    port: u16,
}

fn parse_algorithm(algo_str: &str) -> Algorithm {
    match algo_str.to_lowercase().as_str() {
        "territorial" => Algorithm::Rhea(RheaConfig {
            opponent_model: OpponentModel::Territorial,
            evolutions: 2,
            population_size: 10,
        }),
        "negamax" => Algorithm::Negamax,
        _ => {
            eprintln!("Unknown algorithm '{}', defaulting to territorial", algo_str);
            Algorithm::Rhea(RheaConfig {
                opponent_model: OpponentModel::Territorial,
                evolutions: 200,
                population_size: 50,
            })
        }
    }
}

struct DHState {
    games: Mutex<HashMap<String, AlgorithmInstance>>,
    algorithm_config: Algorithm,
}

async fn info(context: web::Data<DHState>) -> impl Responder {
    let (color, head, tail) = match context.algorithm_config {
        Algorithm::Rhea(ref config) => match config.opponent_model {
            OpponentModel::Territorial => (
                "#73937E".to_string(),  // cambridge blue
                "space-helmet".to_string(),
                "mlh-gene".to_string(),
            ),
            _ => (
                "#888888".to_string(),  // gray fallback
                "default".to_string(),
                "default".to_string(),
            ),
        },
        Algorithm::Negamax => (
            "#FF6B35".to_string(),  // orange-red
            "evil".to_string(),
            "bolt".to_string(),
        ),
    };

    HttpResponse::Ok().json(SnakeInfo {
        color,
        head,
        tail,
    })
}

async fn get_move(context: web::Data<DHState>, state: web::Json<MoveRequest>) -> impl Responder {
    println!("Turn: {:?}", state.turn);

    let game = battlesnake::request_to_game(&state.0);
    let mut all_games = context.games.lock().expect("Failed to get games lock?");

    let mut algo_instance: AlgorithmInstance = match all_games.get(&state.game.id) {
        Some(inst) => inst.clone().update_game(game),
        None => AlgorithmInstance::create(
            context.algorithm_config,
            game,
            state.you.id.clone()
        ),
    };

    let best_move = algo_instance.get_move();

    all_games.insert(state.game.id.clone(), algo_instance);
    println!("Response: {:?}", best_move);

    HttpResponse::Ok().json(MoveResponse {
        r#move: battlesnake::direction_to_string(best_move),
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    env_logger::init_from_env(env_logger::Env::new().default_filter_or("debug"));

    let algorithm = parse_algorithm(&args.algorithm);

    println!("Starting Dancing Horizons server");
    println!("  Algorithm: {:?}", algorithm);
    println!("  Port: {}", args.port);

    let context = web::Data::new(DHState {
        games: Mutex::new(HashMap::new()),
        algorithm_config: algorithm,
    });

    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Logger::default())
            .app_data(context.clone())
            .route("/move", web::post().to(get_move))
            .route("/", web::get().to(info))
    })
    .bind(("127.0.0.1", args.port))?
    .run()
    .await
}
