# Run solo game with territorial RHEA (default)
solo:
	./battlesnake play -W 11 -H 11 --name "dancing-horizons" --url http://localhost:8080 -g solo -v

# Run multi-player game (expects two servers already running)
multi:
	./battlesnake play -W 11 -H 11 --name "dancing-horizons" --url http://localhost:8080 --name "heuy" --url http://localhost:8081 -v -g standard

# Start territorial RHEA server on port 8080
server-territorial:
	cargo run -- --algorithm territorial --port 8080

# Start negamax server on port 8081
server-negamax:
	cargo run -- --algorithm negamax --port 8081

# Battle: territorial vs negamax
# Requires two servers running (use server-territorial and server-negamax in separate terminals)
battle:
	./battlesnake play -W 11 -H 11 \
		--name "Territorial" --url http://localhost:8080 \
		--name "Negamax" --url http://localhost:8081 \
		-v -g standard
