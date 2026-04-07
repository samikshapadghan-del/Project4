function createVehicles(laneClass, count) {
    const lane = document.querySelector("." + laneClass);
    lane.innerHTML = "";

    for (let i = 0; i < count; i++) {
        let v = document.createElement("div");
        v.classList.add("vehicle");

        v.style.top = (i * 25) + "px";
        v.style.left = (i * 25) + "px";

        lane.appendChild(v);
    }
}

function updateTraffic() {

    let lanes = {
        north: Math.floor(Math.random() * 10),
        south: Math.floor(Math.random() * 10),
        east: Math.floor(Math.random() * 10),
        west: Math.floor(Math.random() * 10)
    };

    let total = 0;

    for (let lane in lanes) {
        let value = lanes[lane];
        total += value;

        let laneDiv = document.getElementById(lane);
        laneDiv.querySelector(".count").innerText = value + " vehicles";

        let percent = (value / 10) * 100;
        laneDiv.querySelector(".fill").style.width = percent + "%";
    }

    let greenLane = Object.keys(lanes).reduce((a, b) => lanes[a] > lanes[b] ? a : b);
    let bestLane = Object.keys(lanes).reduce((a, b) => lanes[a] < lanes[b] ? a : b);

    document.getElementById("greenLane").innerText = greenLane.toUpperCase();
    document.getElementById("bestLane").innerText = bestLane.toUpperCase();
    document.getElementById("total").innerText = total;

    createVehicles("northLane", lanes.north);
    createVehicles("southLane", lanes.south);
    createVehicles("eastLane", lanes.east);
    createVehicles("westLane", lanes.west);

    document.querySelectorAll(".vehicle").forEach(v => {
        v.classList.remove("moveNorth", "moveSouth", "moveEast", "moveWest");
    });

    if (greenLane === "north") {
        document.querySelectorAll(".northLane .vehicle").forEach(v => v.classList.add("moveNorth"));
    }
    if (greenLane === "south") {
        document.querySelectorAll(".southLane .vehicle").forEach(v => v.classList.add("moveSouth"));
    }
    if (greenLane === "east") {
        document.querySelectorAll(".eastLane .vehicle").forEach(v => v.classList.add("moveEast"));
    }
    if (greenLane === "west") {
        document.querySelectorAll(".westLane .vehicle").forEach(v => v.classList.add("moveWest"));
    }
}

setInterval(updateTraffic, 3000);