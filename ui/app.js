/**
 * Orbital Inspector - Mission Designer 3D Application
 * 
 * Features:
 * - Real MPC simulation via WebSocket connection
 * - Three.js 3D visualization with live physics data
 * - Mission template loading
 * - Playback controls
 */

// ============================================
// State Management
// ============================================

const state = {
    waypoints: [],
    selectedWaypoint: null,
    inspectorStart: { x: 5, y: 0, z: 0 },
    approachSpeed: 0.05,
    isSimulating: false,
    isPaused: false,
    playbackSpeed: 1,
    followCamera: false,
    simulationTime: 0,
    currentWaypointIndex: 0,
    connected: false,
};

// WebSocket connection
let ws = null;
const WS_URL = 'ws://localhost:8765';

// ============================================
// Mission Templates
// ============================================

const MISSION_TEMPLATES = {
    flyby: {
        name: "Flyby Inspection",
        start: { x: 6, y: 0, z: 0 },
        waypoints: [
            { x: 4.5, y: 0, z: 0, name: "Approach" },
            { x: 3.5, y: 2, z: 0, name: "Flyby Port" },
            { x: 3.5, y: -2, z: 0, name: "Flyby Starboard" },
            { x: 6, y: 0, z: 0, name: "Return" },
        ]
    },
    orbit: {
        name: "Circumnavigation",
        start: { x: 5, y: 0, z: 0 },
        waypoints: [
            { x: 5, y: 0, z: 0, name: "0°" },
            { x: 3.54, y: 3.54, z: 0, name: "45°" },
            { x: 0, y: 5, z: 0, name: "90°" },
            { x: -3.54, y: 3.54, z: 0, name: "135°" },
            { x: -5, y: 0, z: 0, name: "180°" },
            { x: -3.54, y: -3.54, z: 0, name: "225°" },
            { x: 0, y: -5, z: 0, name: "270°" },
            { x: 3.54, y: -3.54, z: 0, name: "315°" },
            { x: 5, y: 0, z: 0, name: "360°" },
        ]
    },
    station: {
        name: "Station Keeping",
        start: { x: 5, y: 0, z: 0 },
        waypoints: [
            { x: 5, y: 0, z: 0, name: "Hold Position" },
        ]
    },
    inspection: {
        name: "6-Point Inspection",
        start: { x: 5, y: 5, z: 0 },
        waypoints: [
            { x: 3, y: 0, z: 0, name: "+X Face" },
            { x: 0, y: 3, z: 0, name: "+Y Face" },
            { x: 0, y: 0, z: 3, name: "+Z Face" },
            { x: -3, y: 0, z: 0, name: "-X Face" },
            { x: 0, y: -3, z: 0, name: "-Y Face" },
            { x: 0, y: 0, z: -3, name: "-Z Face" },
            { x: 5, y: 5, z: 0, name: "Return" },
        ]
    }
};

// ============================================
// Three.js Scene Setup
// ============================================

let scene, camera, renderer, controls;
let targetSatellite, inspectorSatellite;
let waypointMeshes = [];
let trajectoryLine, trailLine;
let raycaster, mouse;
let trailPositions = [];

// Thruster visualization
let thrusterFlames = [];

function initScene() {
    const container = document.getElementById('visualization');
    const canvas = document.getElementById('canvas3d');
    
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050510);
    
    camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.set(12, 12, 12);
    camera.lookAt(0, 0, 0);
    
    renderer = new THREE.WebGLRenderer({ 
        canvas: canvas, 
        antialias: true,
        alpha: true 
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 50;
    
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    
    createStarfield();
    createGrid();
    createTargetSatellite();
    createInspectorSatellite();
    createKeepOutZone();
    createTrajectoryLine();
    createTrailLine();
    addLights();
    
    window.addEventListener('resize', onResize);
    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    
    animate();
}

function createStarfield() {
    const geometry = new THREE.BufferGeometry();
    const vertices = [];
    
    for (let i = 0; i < 2000; i++) {
        vertices.push(
            (Math.random() - 0.5) * 200,
            (Math.random() - 0.5) * 200,
            (Math.random() - 0.5) * 200
        );
    }
    
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    
    const material = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 0.5,
        transparent: true,
        opacity: 0.6,
    });
    
    const stars = new THREE.Points(geometry, material);
    scene.add(stars);
}

function createGrid() {
    const gridHelper = new THREE.GridHelper(20, 20, 0x1a2340, 0x0a0e17);
    gridHelper.position.y = -3;
    scene.add(gridHelper);
}

function createTargetSatellite() {
    const bodyGeom = new THREE.BoxGeometry(1, 1, 1);
    const bodyMat = new THREE.MeshPhongMaterial({ 
        color: 0x4a5568,
        emissive: 0x1a1a2e,
        shininess: 50,
    });
    targetSatellite = new THREE.Mesh(bodyGeom, bodyMat);
    targetSatellite.name = 'target';
    scene.add(targetSatellite);
    
    const panelGeom = new THREE.BoxGeometry(2, 0.1, 0.6);
    const panelMat = new THREE.MeshPhongMaterial({ 
        color: 0x2d3748,
        emissive: 0x1a237e,
        shininess: 80,
    });
    
    const leftPanel = new THREE.Mesh(panelGeom, panelMat);
    leftPanel.position.set(-1.5, 0, 0);
    targetSatellite.add(leftPanel);
    
    const rightPanel = new THREE.Mesh(panelGeom, panelMat);
    rightPanel.position.set(1.5, 0, 0);
    targetSatellite.add(rightPanel);
}

function createInspectorSatellite() {
    const geometry = new THREE.BoxGeometry(0.3, 0.3, 0.3);
    const material = new THREE.MeshPhongMaterial({ 
        color: 0x00d4ff,
        emissive: 0x003344,
        shininess: 100,
    });
    
    inspectorSatellite = new THREE.Mesh(geometry, material);
    inspectorSatellite.position.set(
        state.inspectorStart.x,
        state.inspectorStart.y,
        state.inspectorStart.z
    );
    inspectorSatellite.name = 'inspector';
    scene.add(inspectorSatellite);
    
    // Glow
    const glowGeom = new THREE.SphereGeometry(0.4, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0x00d4ff,
        transparent: true,
        opacity: 0.2,
    });
    const glow = new THREE.Mesh(glowGeom, glowMat);
    inspectorSatellite.add(glow);
    
    // Create thruster flame indicators (6 directions)
    const flameGeom = new THREE.ConeGeometry(0.08, 0.3, 8);
    const flameMat = new THREE.MeshBasicMaterial({ 
        color: 0xff6600, 
        transparent: true,
        opacity: 0.8,
    });
    
    const positions = [
        { pos: [0.25, 0, 0], rot: [0, 0, -Math.PI/2] },   // +X
        { pos: [-0.25, 0, 0], rot: [0, 0, Math.PI/2] },   // -X
        { pos: [0, 0.25, 0], rot: [0, 0, Math.PI] },      // +Y
        { pos: [0, -0.25, 0], rot: [0, 0, 0] },           // -Y
        { pos: [0, 0, 0.25], rot: [Math.PI/2, 0, 0] },    // +Z
        { pos: [0, 0, -0.25], rot: [-Math.PI/2, 0, 0] },  // -Z
    ];
    
    positions.forEach((p, i) => {
        const flame = new THREE.Mesh(flameGeom, flameMat.clone());
        flame.position.set(...p.pos);
        flame.rotation.set(...p.rot);
        flame.visible = false;
        inspectorSatellite.add(flame);
        thrusterFlames.push(flame);
    });
}

function createKeepOutZone() {
    const geometry = new THREE.SphereGeometry(2, 32, 32);
    const material = new THREE.MeshBasicMaterial({
        color: 0xff4444,
        transparent: true,
        opacity: 0.08,
        wireframe: true,
    });
    const keepOut = new THREE.Mesh(geometry, material);
    scene.add(keepOut);
}

function createTrajectoryLine() {
    const material = new THREE.LineBasicMaterial({ 
        color: 0x00d4ff, 
        transparent: true,
        opacity: 0.5,
    });
    const geometry = new THREE.BufferGeometry();
    trajectoryLine = new THREE.Line(geometry, material);
    scene.add(trajectoryLine);
}

function createTrailLine() {
    const material = new THREE.LineBasicMaterial({ 
        color: 0x7c3aed, 
        transparent: true,
        opacity: 0.8,
    });
    const geometry = new THREE.BufferGeometry();
    trailLine = new THREE.Line(geometry, material);
    scene.add(trailLine);
}

function addLights() {
    scene.add(new THREE.AmbientLight(0x404060, 0.5));
    
    const keyLight = new THREE.DirectionalLight(0xffffff, 1);
    keyLight.position.set(10, 10, 10);
    scene.add(keyLight);
    
    const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3);
    fillLight.position.set(-10, 5, -10);
    scene.add(fillLight);
}

// ============================================
// WebSocket Connection
// ============================================

function connectWebSocket() {
    updateConnectionStatus('connecting');
    
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        state.connected = true;
        updateConnectionStatus('connected');
        console.log('Connected to simulation server');
    };
    
    ws.onclose = () => {
        state.connected = false;
        updateConnectionStatus('offline');
        console.log('Disconnected from simulation server');
        // Try to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('error');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'state':
            updateFromPhysics(data);
            break;
        case 'mission_complete':
            onMissionComplete(data);
            break;
        case 'paused':
            state.isPaused = true;
            updatePlayButton();
            break;
        case 'resumed':
            state.isPaused = false;
            updatePlayButton();
            break;
        case 'stopped':
            state.isSimulating = false;
            onSimulationEnd();
            break;
    }
}

function updateFromPhysics(data) {
    // Update inspector position from real physics
    const pos = data.position;
    inspectorSatellite.position.set(pos[0], pos[1], pos[2]);
    
    // Update quaternion (attitude)
    const quat = data.quaternion;
    inspectorSatellite.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
    
    // Record trail
    trailPositions.push(new THREE.Vector3(pos[0], pos[1], pos[2]));
    if (trailPositions.length > 1000) trailPositions.shift();
    updateTrail();
    
    // Update thruster visualization
    updateThrusters(data.control);
    
    // Update telemetry
    state.simulationTime = data.time;
    state.currentWaypointIndex = data.waypoint_idx;
    
    const vel = data.velocity;
    const speed = Math.sqrt(vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]);
    
    document.getElementById('telPos').textContent = 
        `(${pos[0].toFixed(2)}, ${pos[1].toFixed(2)}, ${pos[2].toFixed(2)})`;
    document.getElementById('telVel').textContent = `${speed.toFixed(3)} m/s`;
    document.getElementById('telDist').textContent = 
        `${Math.sqrt(pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]).toFixed(2)} m`;
    document.getElementById('telWaypoint').textContent = 
        `${data.waypoint_idx} / ${data.total_waypoints}`;
    document.getElementById('telTime').textContent = `${data.time.toFixed(1)}s`;
    
    const progress = data.total_waypoints > 0 
        ? (data.waypoint_idx / data.total_waypoints) * 100 
        : 0;
    document.getElementById('progressFill').style.width = `${progress}%`;
    
    // Follow camera
    if (state.followCamera) {
        camera.position.set(pos[0] + 5, pos[1] + 5, pos[2] + 5);
        camera.lookAt(pos[0], pos[1], pos[2]);
    }
}

function updateThrusters(control) {
    if (!control) return;

    const thrusterThreshold = 0.01;

    // Handle 12-thruster model (SatelliteMPCLinearizedSimulation)
    if (control.length === 12) {
        // Mapping based on satellite_3d.xml positions:
        // Flame 0 (+X face): Thrusters 0, 1 (Indices 1,2 in XML)
        // Flame 1 (-X face): Thrusters 4, 5 (Indices 5,6 in XML)
        // Flame 2 (+Y face): Thrusters 6, 7 (Indices 7,8 in XML)
        // Flame 3 (-Y face): Thrusters 2, 3 (Indices 3,4 in XML)
        // Flame 4 (+Z face): Thrusters 8, 9 (Indices 9,10 in XML)
        // Flame 5 (-Z face): Thrusters 10, 11 (Indices 11,12 in XML)
        
        const flameMap = [
            Math.max(control[0], control[1]),   // +X
            Math.max(control[4], control[5]),   // -X
            Math.max(control[6], control[7]),   // +Y
            Math.max(control[2], control[3]),   // -Y
            Math.max(control[8], control[9]),   // +Z
            Math.max(control[10], control[11])  // -Z
        ];

        for (let i = 0; i < 6; i++) {
            const thrust = flameMap[i];
            if (thrust > thrusterThreshold) {
                thrusterFlames[i].visible = true;
                thrusterFlames[i].scale.setScalar(0.5 + thrust * 2);
            } else {
                thrusterFlames[i].visible = false;
            }
        }
        return;
    }

    // Handle 9-element RW model (Legacy / ReactionWheelMPCController)
    // Control format: [rw1, rw2, rw3, thr1, thr2, thr3, thr4, thr5, thr6]
    // Thrusters are indices 3-8
    if (control.length >= 9) {
        for (let i = 0; i < 6; i++) {
            const thrust = Math.abs(control[3 + i]);
            if (thrust > thrusterThreshold) {
                thrusterFlames[i].visible = true;
                thrusterFlames[i].scale.setScalar(0.5 + thrust * 2);
            } else {
                thrusterFlames[i].visible = false;
            }
        }
    }
}

function updateConnectionStatus(status) {
    const el = document.getElementById('connectionStatus');
    switch (status) {
        case 'connected':
            el.textContent = '🟢 Connected';
            el.className = 'status-value connected';
            break;
        case 'connecting':
            el.textContent = '🟡 Connecting...';
            el.className = 'status-value connecting';
            break;
        case 'offline':
            el.textContent = '⚪ Offline';
            el.className = 'status-value';
            break;
        case 'error':
            el.textContent = '🔴 Error';
            el.className = 'status-value error';
            break;
    }
}

// ============================================
// Simulation Control
// ============================================

function startSimulation() {
    if (state.waypoints.length === 0) {
        alert('Add waypoints first!');
        return;
    }
    
    if (!state.connected) {
        alert('Not connected to simulation server.\n\nRun: python3 scripts/simulation_server.py');
        return;
    }
    
    if (state.isSimulating && !state.isPaused) {
        // Pause
        ws.send(JSON.stringify({ type: 'pause' }));
        return;
    }
    
    if (state.isPaused) {
        // Resume
        ws.send(JSON.stringify({ type: 'resume' }));
        return;
    }
    
    // Start new simulation
    state.isSimulating = true;
    state.isPaused = false;
    trailPositions = [];
    
    ws.send(JSON.stringify({
        type: 'start',
        waypoints: state.waypoints,
        start_position: [state.inspectorStart.x, state.inspectorStart.y, state.inspectorStart.z],
        speed: state.playbackSpeed,
    }));
    
    document.getElementById('playPauseBtn').textContent = '⏸';
    document.getElementById('playPauseBtn').classList.add('playing');
    document.getElementById('liveIndicator').textContent = '🟢';
    document.getElementById('liveIndicator').classList.add('active');
}

function stopSimulation() {
    if (ws && state.connected) {
        ws.send(JSON.stringify({ type: 'stop' }));
    }
    state.isSimulating = false;
    state.isPaused = false;
    trailPositions = [];
    updateTrail();
    onSimulationEnd();
}

function onMissionComplete(data) {
    state.isSimulating = false;
    console.log('Mission complete!', data);
    onSimulationEnd();
    
    // Show completion message
    document.getElementById('telTime').textContent = `${data.time.toFixed(1)}s ✓`;
}

function onSimulationEnd() {
    document.getElementById('playPauseBtn').textContent = '▶';
    document.getElementById('playPauseBtn').classList.remove('playing');
    document.getElementById('liveIndicator').textContent = '⚫';
    document.getElementById('liveIndicator').classList.remove('active');
}

function updatePlayButton() {
    if (state.isPaused) {
        document.getElementById('playPauseBtn').textContent = '▶';
        document.getElementById('playPauseBtn').classList.remove('playing');
    } else {
        document.getElementById('playPauseBtn').textContent = '⏸';
        document.getElementById('playPauseBtn').classList.add('playing');
    }
}

// ============================================
// Waypoint Management
// ============================================

function addWaypoint(x, y, z, name = '') {
    const id = Date.now();
    const waypoint = { id, x, y, z, name: name || `Point ${state.waypoints.length + 1}` };
    state.waypoints.push(waypoint);
    
    const geometry = new THREE.SphereGeometry(0.15, 16, 16);
    const material = new THREE.MeshPhongMaterial({ 
        color: 0x7c3aed,
        emissive: 0x3d1a8a,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.userData.waypointId = id;
    scene.add(mesh);
    waypointMeshes.push(mesh);
    
    updateWaypointList();
    updateTrajectory();
    
    return waypoint;
}

function removeWaypoint(id) {
    const index = state.waypoints.findIndex(w => w.id === id);
    if (index !== -1) {
        state.waypoints.splice(index, 1);
        
        const meshIndex = waypointMeshes.findIndex(m => m.userData.waypointId === id);
        if (meshIndex !== -1) {
            scene.remove(waypointMeshes[meshIndex]);
            waypointMeshes.splice(meshIndex, 1);
        }
        
        updateWaypointList();
        updateTrajectory();
    }
}

function clearWaypoints() {
    state.waypoints = [];
    waypointMeshes.forEach(m => scene.remove(m));
    waypointMeshes = [];
    updateWaypointList();
    updateTrajectory();
}

function updateWaypointList() {
    const listEl = document.getElementById('waypointList');
    
    if (state.waypoints.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state">
                <p>No waypoints yet</p>
                <p class="hint">Click in 3D view or use + Add</p>
            </div>
        `;
        return;
    }
    
    listEl.innerHTML = state.waypoints.map((wp, i) => `
        <div class="waypoint-item" data-id="${wp.id}">
            <span class="waypoint-number">${i + 1}</span>
            <span class="waypoint-coords">${wp.name}</span>
            <button class="waypoint-delete" onclick="removeWaypoint(${wp.id})">✕</button>
        </div>
    `).join('');
}

function updateTrajectory() {
    const points = [];
    
    points.push(new THREE.Vector3(
        state.inspectorStart.x,
        state.inspectorStart.y,
        state.inspectorStart.z
    ));
    
    state.waypoints.forEach(wp => {
        points.push(new THREE.Vector3(wp.x, wp.y, wp.z));
    });
    
    if (points.length > 1) {
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        trajectoryLine.geometry.dispose();
        trajectoryLine.geometry = geometry;
        trajectoryLine.visible = true;
    } else {
        trajectoryLine.visible = false;
    }
}

function updateTrail() {
    if (trailPositions.length < 2) {
        trailLine.visible = false;
        return;
    }
    
    const geometry = new THREE.BufferGeometry().setFromPoints(trailPositions);
    trailLine.geometry.dispose();
    trailLine.geometry = geometry;
    trailLine.visible = true;
}

// ============================================
// Template Loading
// ============================================

function loadTemplate(templateId) {
    const template = MISSION_TEMPLATES[templateId];
    if (!template) return;
    
    clearWaypoints();
    
    state.inspectorStart = { ...template.start };
    document.getElementById('startX').value = template.start.x;
    document.getElementById('startY').value = template.start.y;
    document.getElementById('startZ').value = template.start.z;
    
    inspectorSatellite.position.set(
        template.start.x,
        template.start.y,
        template.start.z
    );
    
    template.waypoints.forEach(wp => {
        addWaypoint(wp.x, wp.y, wp.z, wp.name);
    });
    
    document.getElementById('missionTemplate').value = '';
}

// ============================================
// Event Handlers
// ============================================

function onResize() {
    const container = document.getElementById('visualization');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function onCanvasClick(event) {
    if (state.isSimulating) return;
    
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersection = new THREE.Vector3();
    
    if (raycaster.ray.intersectPlane(plane, intersection)) {
        if (intersection.length() > 2.5) {
            addWaypoint(
                Math.round(intersection.x * 2) / 2,
                0,
                Math.round(intersection.z * 2) / 2
            );
        }
    }
}

function onCanvasMouseMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
    
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersection = new THREE.Vector3();
    
    if (raycaster.ray.intersectPlane(plane, intersection)) {
        document.getElementById('coordDisplay').textContent = 
            `X: ${intersection.x.toFixed(2)} Y: ${intersection.y.toFixed(2)} Z: ${intersection.z.toFixed(2)}`;
    }
}

// ============================================
// Animation Loop
// ============================================

function animate() {
    requestAnimationFrame(animate);
    
    if (!state.followCamera) {
        controls.update();
    }
    
    if (targetSatellite) {
        targetSatellite.rotation.y += 0.001;
    }
    
    renderer.render(scene, camera);
}

// ============================================
// UI Initialization
// ============================================

function initUI() {
    document.getElementById('addWaypointBtn').addEventListener('click', () => {
        const angle = Math.random() * Math.PI * 2;
        const radius = 4 + Math.random() * 3;
        addWaypoint(
            Math.round(Math.cos(angle) * radius * 2) / 2,
            0,
            Math.round(Math.sin(angle) * radius * 2) / 2
        );
    });
    
    const speedSlider = document.getElementById('approachSpeed');
    const speedDisplay = document.getElementById('speedDisplay');
    speedSlider.addEventListener('input', (e) => {
        state.approachSpeed = parseFloat(e.target.value);
        speedDisplay.textContent = `${state.approachSpeed.toFixed(2)} m/s`;
    });
    
    ['startX', 'startY', 'startZ'].forEach(id => {
        document.getElementById(id).addEventListener('change', () => {
            state.inspectorStart.x = parseFloat(document.getElementById('startX').value);
            state.inspectorStart.y = parseFloat(document.getElementById('startY').value);
            state.inspectorStart.z = parseFloat(document.getElementById('startZ').value);
            
            if (inspectorSatellite && !state.isSimulating) {
                inspectorSatellite.position.set(
                    state.inspectorStart.x,
                    state.inspectorStart.y,
                    state.inspectorStart.z
                );
            }
            updateTrajectory();
        });
    });
    
    document.getElementById('resetViewBtn').addEventListener('click', () => {
        camera.position.set(12, 12, 12);
        camera.lookAt(0, 0, 0);
        state.followCamera = false;
        document.getElementById('followBtn').classList.remove('active');
    });
    
    document.getElementById('topViewBtn').addEventListener('click', () => {
        camera.position.set(0, 20, 0.01);
        camera.lookAt(0, 0, 0);
    });
    
    document.getElementById('sideViewBtn').addEventListener('click', () => {
        camera.position.set(20, 0, 0);
        camera.lookAt(0, 0, 0);
    });
    
    document.getElementById('followBtn').addEventListener('click', () => {
        state.followCamera = !state.followCamera;
        document.getElementById('followBtn').classList.toggle('active', state.followCamera);
    });
    
    document.getElementById('runBtn').addEventListener('click', startSimulation);
    document.getElementById('playPauseBtn').addEventListener('click', startSimulation);
    document.getElementById('stopBtn').addEventListener('click', stopSimulation);
    
    const playbackSlider = document.getElementById('playbackSpeed');
    playbackSlider.addEventListener('input', (e) => {
        state.playbackSpeed = parseFloat(e.target.value);
        document.getElementById('speedLabel').textContent = `${state.playbackSpeed}x`;
        if (ws && state.connected) {
            ws.send(JSON.stringify({ type: 'speed', speed: state.playbackSpeed }));
        }
    });
    
    document.getElementById('missionTemplate').addEventListener('change', (e) => {
        if (e.target.value) {
            loadTemplate(e.target.value);
        }
    });
    
    document.getElementById('exportBtn').addEventListener('click', exportMission);
    
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && state.waypoints.length > 0) {
            e.preventDefault();
            startSimulation();
        }
        if (e.code === 'Escape') {
            stopSimulation();
        }
    });
}

function exportMission() {
    const mission = {
        name: "Custom Mission",
        mission_type: "custom",
        start_position: [state.inspectorStart.x, state.inspectorStart.y, state.inspectorStart.z],
        waypoints: state.waypoints.map(wp => ({
            position: [wp.x, wp.y, wp.z],
            name: wp.name,
        })),
        timestamp: new Date().toISOString(),
    };
    
    const blob = new Blob([JSON.stringify(mission, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mission.json';
    a.click();
    
    URL.revokeObjectURL(url);
}

// ============================================
// Initialize
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initScene();
    initUI();
    connectWebSocket();
});
