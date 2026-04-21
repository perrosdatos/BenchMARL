import numpy as np

def test_reward(act=0, delta_eng=0, collide=False):
    stagnation = 0.0
    movement = 0.0
    energy_gain = 0.0
    collision = 0.0
    
    if act == 0:
        if delta_eng <= 0:
            stagnation = 1.0 / 16.0
    else:
        movement = 1.0 / 16.0
        
    energy_gain = (delta_eng / 400.0) / 16.0
    
    if collide:
        collision = (2.0 / 16.0) / 16.0

    r = (
        collision * -32.0 +
        movement * 0.2 +
        energy_gain * 0.8 +
        stagnation * -0.5
    )
    return r

print("Idle, 0 energy:", test_reward(0, 0))
print("Move, -1 energy:", test_reward(1, -1))
print("Move, -1 energy, collide:", test_reward(1, -1, True))
