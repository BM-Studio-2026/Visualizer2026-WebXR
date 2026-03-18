import * as THREE from 'three';
import { VRButton } from 'three/examples/jsm/webxr/VRButton.js';

console.log("JS is running");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202020);

const camera = new THREE.PerspectiveCamera(
  70,
  window.innerWidth / window.innerHeight,
  0.1,
  100
);
camera.position.set(0, 1.6, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;

document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

// Desktop label
const statusDiv = document.createElement('div');
statusDiv.style.position = 'absolute';
statusDiv.style.top = '10px';
statusDiv.style.left = '10px';
statusDiv.style.color = 'white';
statusDiv.style.fontFamily = 'Arial, sans-serif';
statusDiv.style.fontSize = '20px';
statusDiv.style.background = 'rgba(0,0,0,0.5)';
statusDiv.style.padding = '8px 12px';
statusDiv.style.borderRadius = '8px';
document.body.appendChild(statusDiv);

// Axes
const axes = new THREE.AxesHelper(2);
scene.add(axes);

// Light
const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.5);
scene.add(light);

// Cube
const geometry = new THREE.BoxGeometry(1, 1, 1);
const material = new THREE.MeshNormalMaterial();
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

// Matrices
const identityMatrix = new THREE.Matrix4().set(
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 1
);

const scaleMatrix = new THREE.Matrix4().set(
  1.5, 0,   0,   0,
  0,   1,   0,   0,
  0,   0,   0.5, 0,
  0,   0,   0,   1
);

const shearMatrix = new THREE.Matrix4().set(
  1, 0.8, 0, 0,
  0, 1,   0, 0,
  0, 0,   1, 0,
  0, 0,   0, 1
);

const rotationMatrix = new THREE.Matrix4().makeRotationZ(Math.PI / 6);

const matrices = [
  { name: 'Identity', matrix: identityMatrix },
  { name: 'Scale', matrix: scaleMatrix },
  { name: 'Shear', matrix: shearMatrix },
  { name: 'Rotation Z', matrix: rotationMatrix }
];

let currentIndex = 0;

cube.matrixAutoUpdate = false;

function applyMatrixToCube(matrix) {
  const translation = new THREE.Matrix4().makeTranslation(0, 1.6, -2);
  const worldMatrix = new THREE.Matrix4().multiplyMatrices(translation, matrix);
  cube.matrix.copy(worldMatrix);
  statusDiv.textContent = `Current transform: ${matrices[currentIndex].name}`;
}

applyMatrixToCube(matrices[currentIndex].matrix);

// Desktop keyboard support
window.addEventListener('keydown', (event) => {
  if (event.key === '1') currentIndex = 0;
  if (event.key === '2') currentIndex = 1;
  if (event.key === '3') currentIndex = 2;
  if (event.key === '4') currentIndex = 3;

  applyMatrixToCube(matrices[currentIndex].matrix);
});

// Controllers
const controller1 = renderer.xr.getController(0);
scene.add(controller1);

const controller2 = renderer.xr.getController(1);
scene.add(controller2);

function addControllerRay(controller) {
  const rayGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -1)
  ]);
  const rayMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
  const line = new THREE.Line(rayGeometry, rayMaterial);
  line.scale.z = 2;
  controller.add(line);
}

addControllerRay(controller1);
addControllerRay(controller2);

function onSelectStart() {
  currentIndex = (currentIndex + 1) % matrices.length;
  applyMatrixToCube(matrices[currentIndex].matrix);
  console.log('Switched to:', matrices[currentIndex].name);
}

controller1.addEventListener('selectstart', onSelectStart);
controller2.addEventListener('selectstart', onSelectStart);

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Render loop
renderer.setAnimationLoop(() => {
  renderer.render(scene, camera);
});