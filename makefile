solo:
	./battlesnake play -W 11 -H 11 --name "dancing-horizons" --url http://localhost:9999 -g solo -v --browser

duel:
	./battlesnake play -W 11 -H 11 --name "dancing-horizons" --url http://localhost:9999 --name "eater" --url http://localhost:8080 -v --browser
