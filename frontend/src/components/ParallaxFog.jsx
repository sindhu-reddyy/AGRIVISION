import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Cloud } from '@react-three/drei';

const ParallaxFog = () => {
    const group = useRef();

    useFrame((state) => {
        // Gentle rotation or movement
        if (group.current) {
            group.current.rotation.y = state.clock.getElapsedTime() * 0.02;
        }
    });

    return (
        <group ref={group}>
            <Cloud opacity={0.5} speed={0.4} width={10} depth={1.5} segments={20} position={[0, -2, -5]} color="#a0e0a0" />
            <Cloud opacity={0.3} speed={0.2} width={10} depth={2} segments={10} position={[5, 2, -10]} color="#c0ffc0" />
            <Cloud opacity={0.4} speed={0.6} width={15} depth={3} segments={15} position={[-5, 0, -8]} color="#d0ffd0" />
        </group>
    );
};

export default ParallaxFog;
