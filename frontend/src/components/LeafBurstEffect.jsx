import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

const LeafBurstEffect = ({ active }) => {
    const count = 200;
    const mesh = useRef();
    const dummy = useMemo(() => new THREE.Object3D(), []);

    const particles = useMemo(() => {
        const temp = [];
        for (let i = 0; i < count; i++) {
            const angle = Math.random() * Math.PI * 2;
            const r = Math.random() * 2; // radius
            const speed = 0.05 + Math.random() * 0.1;
            temp.push({
                x: 0, y: 0, z: 0,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                vz: (Math.random() - 0.5) * speed,
                life: 1.0
            });
        }
        return temp;
    }, []);

    useFrame(() => {
        if (!mesh.current || !active) return;

        particles.forEach((particle, i) => {
            if (active) {
                particle.x += particle.vx;
                particle.y += particle.vy;
                particle.z += particle.vz;
                particle.life -= 0.01;
            }

            dummy.position.set(particle.x, particle.y, particle.z);
            const scale = Math.max(0, particle.life);
            dummy.scale.set(scale, scale, scale);
            dummy.lookAt(0, 0, 0);
            dummy.updateMatrix();
            mesh.current.setMatrixAt(i, dummy.matrix);
        });
        mesh.current.instanceMatrix.needsUpdate = true;
    });

    if (!active) return null;

    return (
        <instancedMesh ref={mesh} args={[null, null, count]}>
            <coneGeometry args={[0.1, 0.2, 4]} />
            <meshBasicMaterial color="#a8ffaa" />
        </instancedMesh>
    );
};

export default LeafBurstEffect;
