# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dancing Horizons is a Battlesnake AI implementation using the Rolling Horizon Evolutionary Algorithm (RHEA), based on the adaptation described in [this paper](https://sbgames.org/sbgames2018/files/papers/ComputacaoFull/187244.pdf).

## Architecture

The codebase is structured around several key components:

- **bitboard.rs**: Implements efficient bitboard representation for the game board using u128 integers. Handles snake positions, movements, and collision detection on an 11x11 grid.
- **rhea.rs**: Core RHEA implementation with genetic algorithm components including population management, mutation, tournament selection, and fitness evaluation.
- **battlesnake.rs**: API integration layer that converts between Battlesnake JSON protocol and internal game representation.
- **main.rs**: Actix-web HTTP server that maintains game state across turns and executes 400 evolution cycles per move decision.

The system uses bitboard operations for fast game state simulation and runs evolutionary search to find optimal move sequences.

## Development Commands

**Build and run:**
```bash
cargo run
```

**Testing with battlesnake CLI:**
```bash
# Solo game
make solo

# Multi-player game  
make multi
```

**Build:**
```bash
cargo build
```

**Check code:**
```bash
cargo check
cargo clippy
```

## Key Implementation Details

- Server runs on localhost:8080
- Game state is maintained in memory using HashMap keyed by game ID
- RHEA runs 400 evolution cycles per move (configurable in main.rs:38)
- Population size: 50 candidates with genotype length of 20 moves
- Mutation chance: 30% with tournament selection (size 3)
- Bitboard uses u128 to represent 11x11 grid with efficient bit manipulation for movement validation