import React, { useMemo } from 'react';
import * as THREE from 'three';

const Leaf = ({ position, rotation, scale, color }) => {
    return (
        <group position={position} rotation={rotation} scale={scale}>
            {/* Simple leaf shape using a scaled sphere or cone */}
            <mesh position={[0, 0.5, 0]} scale={[0.2, 1, 0.05]}>
                <sphereGeometry args={[1, 8, 8]} />
                <meshStandardMaterial color={color} roughness={0.6} side={THREE.DoubleSide} />
            </mesh>
        </group>
    );
};

const ThreeDPlant = ({ scale = 1, color = "#16a34a" }) => {
    // Procedurally generate a fern-like structure
    const structure = useMemo(() => {
        const stems = [];
        const numFronds = 5 + Math.floor(Math.random() * 3); // 5-8 fronds

        for (let i = 0; i < numFronds; i++) {
            const frondAngle = (i / numFronds) * Math.PI * 2;
            const curveStrength = 0.5 + Math.random() * 0.5;
            const height = 2 + Math.random();

            // Frond stem points
            stems.push({
                type: 'stem',
                angle: frondAngle,
                curve: curveStrength,
                height: height
            });
        }
        return stems;
    }, []);

    return (
        <group scale={[scale, scale, scale]}>
            {structure.map((frond, i) => (
                <group key={i} rotation={[0, frond.angle, 0]}>
                    {/* Main Stem Curve - approximated by rotating segments or just one bent tube? 
                 Let's use a simple bent cylinder logic by stacking them or utilizing a curve object if we want.
                 For simplicity/performance in R3F without heavy geometry, let's use a group of objects rotated outward.
             */}
                    <group rotation={[Math.PI / 4, 0, 0]}> {/* Angle out from center */}
                        <mesh position={[0, frond.height / 2, 0]}>
                            <cylinderGeometry args={[0.02, 0.05, frond.height, 4]} />
                            <meshStandardMaterial color={color} roughness={0.8} />
                        </mesh>

                        {/* Leaves along the stem */}
                        {Array.from({ length: 8 }).map((_, j) => {
                            const t = j / 8; // 0 to 1 along stem
                            const y = t * frond.height;
                            const leafScale = (1 - t) * 0.8 + 0.2; // Taper at top
                            return (
                                <group key={j} position={[0, y, 0]}>
                                    <Leaf position={[0.1, 0, 0]} rotation={[0, 0, Math.PI / 3]} scale={[leafScale, leafScale, leafScale]} color={color} />
                                    <Leaf position={[-0.1, 0, 0]} rotation={[0, 0, -Math.PI / 3]} scale={[leafScale, leafScale, leafScale]} color={color} />
                                </group>
                            );
                        })}
                    </group>
                </group>
            ))}
        </group>
    );
};

export default ThreeDPlant;
