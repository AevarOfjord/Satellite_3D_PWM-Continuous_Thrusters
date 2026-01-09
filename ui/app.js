/**
 * Orbital Inspector - Mission Designer 3D Application
 * 
 * Three.js scene with:
 * - Target satellite at origin
 * - Inspector satellite (movable)
 * - Waypoint markers
 * - Interactive controls
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
};

// ============================================
// Three.js Scene Setup
// ============================================

let scene, camera, renderer, controls;
let targetSatellite, inspectorSatellite;
let waypointMeshes = [];
let trajectoryLine;
let raycaster, mouse;

function initScene() {
    const container = document.getElementById('visualization');
    const canvas = document.getElementById('canvas3d');
    
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x050510);
    
    // Camera
    camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.set(12, 12, 12);
    camera.lookAt(0, 0, 0);
    
    // Renderer
    renderer = new THREE.WebGLRenderer({ 
        canvas: canvas, 
        antialias: true,
        alpha: true 
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 50;
    
    // Raycaster for click detection
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();
    
    // Add objects
    createStarfield();
    createGrid();
    createTargetSatellite();
    createInspectorSatellite();
    createKeepOutZone();
    createTrajectoryLine();
    addLights();
    
    // Event listeners
    window.addEventListener('resize', onResize);
    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    
    // Start animation loop
    animate();
}

// ============================================
// 3D Objects
// ============================================

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
    // Main body
    const bodyGeom = new THREE.BoxGeometry(1, 1, 1);
    const bodyMat = new THREE.MeshPhongMaterial({ 
        color: 0x4a5568,
        emissive: 0x1a1a2e,
        shininess: 50,
    });
    targetSatellite = new THREE.Mesh(bodyGeom, bodyMat);
    targetSatellite.name = 'target';
    scene.add(targetSatellite);
    
    // Solar panels
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
    
    // Antenna
    const antennaGeom = new THREE.CylinderGeometry(0.05, 0.05, 0.8, 8);
    const antennaMat = new THREE.MeshPhongMaterial({ color: 0xcccccc });
    const antenna = new THREE.Mesh(antennaGeom, antennaMat);
    antenna.position.set(0, 0.9, 0);
    targetSatellite.add(antenna);
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
    
    // Add glow effect
    const glowGeom = new THREE.SphereGeometry(0.4, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0x00d4ff,
        transparent: true,
        opacity: 0.2,
    });
    const glow = new THREE.Mesh(glowGeom, glowMat);
    inspectorSatellite.add(glow);
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
    keepOut.name = 'keepout';
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

function addLights() {
    // Ambient
    const ambient = new THREE.AmbientLight(0x404060, 0.5);
    scene.add(ambient);
    
    // Key light
    const keyLight = new THREE.DirectionalLight(0xffffff, 1);
    keyLight.position.set(10, 10, 10);
    scene.add(keyLight);
    
    // Fill light
    const fillLight = new THREE.DirectionalLight(0x4488ff, 0.3);
    fillLight.position.set(-10, 5, -10);
    scene.add(fillLight);
    
    // Accent light for inspector
    const accentLight = new THREE.PointLight(0x00d4ff, 0.5, 10);
    accentLight.position.set(5, 2, 5);
    scene.add(accentLight);
}

// ============================================
// Waypoint Management
// ============================================

function addWaypoint(x, y, z) {
    const id = Date.now();
    const waypoint = { id, x, y, z };
    state.waypoints.push(waypoint);
    
    // Create 3D marker
    const geometry = new THREE.SphereGeometry(0.15, 16, 16);
    const material = new THREE.MeshPhongMaterial({ 
        color: 0x7c3aed,
        emissive: 0x3d1a8a,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.userData.waypointId = id;
    mesh.name = 'waypoint';
    scene.add(mesh);
    waypointMeshes.push(mesh);
    
    // Update UI and trajectory
    updateWaypointList();
    updateTrajectory();
    
    return waypoint;
}

function removeWaypoint(id) {
    const index = state.waypoints.findIndex(w => w.id === id);
    if (index !== -1) {
        state.waypoints.splice(index, 1);
        
        // Remove 3D mesh
        const meshIndex = waypointMeshes.findIndex(m => m.userData.waypointId === id);
        if (meshIndex !== -1) {
            scene.remove(waypointMeshes[meshIndex]);
            waypointMeshes.splice(meshIndex, 1);
        }
        
        updateWaypointList();
        updateTrajectory();
    }
}

function updateWaypointList() {
    const listEl = document.getElementById('waypointList');
    
    if (state.waypoints.length === 0) {
        listEl.innerHTML = `
            <div class="empty-state">
                <p>No waypoints yet</p>
                <p class="hint">Click in the 3D view or use + Add</p>
            </div>
        `;
        return;
    }
    
    listEl.innerHTML = state.waypoints.map((wp, i) => `
        <div class="waypoint-item" data-id="${wp.id}">
            <span class="waypoint-number">${i + 1}</span>
            <span class="waypoint-coords">(${wp.x.toFixed(1)}, ${wp.y.toFixed(1)}, ${wp.z.toFixed(1)})</span>
            <button class="waypoint-delete" onclick="removeWaypoint(${wp.id})">✕</button>
        </div>
    `).join('');
}

function updateTrajectory() {
    const points = [];
    
    // Start from inspector position
    points.push(new THREE.Vector3(
        state.inspectorStart.x,
        state.inspectorStart.y,
        state.inspectorStart.z
    ));
    
    // Add waypoints
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
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    raycaster.setFromCamera(mouse, camera);
    
    // Check for click on ground plane (y = 0)
    const planeNormal = new THREE.Vector3(0, 1, 0);
    const plane = new THREE.Plane(planeNormal, 0);
    const intersection = new THREE.Vector3();
    
    if (raycaster.ray.intersectPlane(plane, intersection)) {
        // Check if position is valid (outside keep-out zone)
        const dist = intersection.length();
        if (dist > 2.5) {
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
    
    const planeNormal = new THREE.Vector3(0, 1, 0);
    const plane = new THREE.Plane(planeNormal, 0);
    const intersection = new THREE.Vector3();
    
    if (raycaster.ray.intersectPlane(plane, intersection)) {
        document.getElementById('coordDisplay').textContent = 
            `X: ${intersection.x.toFixed(2)} Y: ${intersection.y.toFixed(2)} Z: ${intersection.z.toFixed(2)}`;
    }
}

// ============================================
// Animation
// ============================================

function animate() {
    requestAnimationFrame(animate);
    
    controls.update();
    
    // Rotate target satellite slowly
    if (targetSatellite) {
        targetSatellite.rotation.y += 0.001;
    }
    
    renderer.render(scene, camera);
}

// ============================================
// UI Controls
// ============================================

function initUI() {
    // Add waypoint button
    document.getElementById('addWaypointBtn').addEventListener('click', () => {
        // Add at a random position 
        const angle = Math.random() * Math.PI * 2;
        const radius = 4 + Math.random() * 3;
        addWaypoint(
            Math.round(Math.cos(angle) * radius * 2) / 2,
            0,
            Math.round(Math.sin(angle) * radius * 2) / 2
        );
    });
    
    // Speed slider
    const speedSlider = document.getElementById('approachSpeed');
    const speedDisplay = document.getElementById('speedDisplay');
    speedSlider.addEventListener('input', (e) => {
        state.approachSpeed = parseFloat(e.target.value);
        speedDisplay.textContent = `${state.approachSpeed.toFixed(2)} m/s`;
    });
    
    // Start position inputs
    ['startX', 'startY', 'startZ'].forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('change', () => {
            state.inspectorStart.x = parseFloat(document.getElementById('startX').value);
            state.inspectorStart.y = parseFloat(document.getElementById('startY').value);
            state.inspectorStart.z = parseFloat(document.getElementById('startZ').value);
            
            if (inspectorSatellite) {
                inspectorSatellite.position.set(
                    state.inspectorStart.x,
                    state.inspectorStart.y,
                    state.inspectorStart.z
                );
            }
            updateTrajectory();
        });
    });
    
    // View buttons
    document.getElementById('resetViewBtn').addEventListener('click', () => {
        camera.position.set(12, 12, 12);
        camera.lookAt(0, 0, 0);
    });
    
    document.getElementById('topViewBtn').addEventListener('click', () => {
        camera.position.set(0, 20, 0.01);
        camera.lookAt(0, 0, 0);
    });
    
    document.getElementById('sideViewBtn').addEventListener('click', () => {
        camera.position.set(20, 0, 0);
        camera.lookAt(0, 0, 0);
    });
    
    // Run simulation
    document.getElementById('runBtn').addEventListener('click', runSimulation);
    
    // Export
    document.getElementById('exportBtn').addEventListener('click', exportMission);
}

// ============================================
// Simulation
// ============================================

async function runSimulation() {
    const statusEl = document.getElementById('simStatus');
    statusEl.textContent = 'Running...';
    statusEl.className = 'status-value connecting';
    
    // Build mission data
    const mission = {
        inspector_start: state.inspectorStart,
        waypoints: state.waypoints,
        approach_speed: state.approachSpeed,
    };
    
    try {
        // For now, just simulate locally
        console.log('Mission data:', mission);
        
        // Animate inspector along trajectory
        await animateInspector(mission);
        
        statusEl.textContent = 'Complete';
        statusEl.className = 'status-value connected';
    } catch (error) {
        console.error('Simulation error:', error);
        statusEl.textContent = 'Error';
        statusEl.className = 'status-value error';
    }
}

async function animateInspector(mission) {
    const points = [
        new THREE.Vector3(
            mission.inspector_start.x,
            mission.inspector_start.y,
            mission.inspector_start.z
        ),
        ...mission.waypoints.map(wp => new THREE.Vector3(wp.x, wp.y, wp.z))
    ];
    
    if (points.length < 2) return;
    
    let currentSegment = 0;
    let t = 0;
    const speed = 0.01;
    
    return new Promise(resolve => {
        function step() {
            if (currentSegment >= points.length - 1) {
                resolve();
                return;
            }
            
            const start = points[currentSegment];
            const end = points[currentSegment + 1];
            
            t += speed;
            
            if (t >= 1) {
                t = 0;
                currentSegment++;
                if (currentSegment >= points.length - 1) {
                    resolve();
                    return;
                }
            }
            
            const pos = new THREE.Vector3().lerpVectors(start, end, t);
            inspectorSatellite.position.copy(pos);
            
            requestAnimationFrame(step);
        }
        
        step();
    });
}

function exportMission() {
    const mission = {
        inspector_start: state.inspectorStart,
        waypoints: state.waypoints,
        approach_speed: state.approachSpeed,
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
    
    // Update connection status (offline for now)
    document.getElementById('connectionStatus').textContent = '⚪ Offline';
});
