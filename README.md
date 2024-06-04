# Double Q-Learning for a Simple Parking Problem

## Sample videos

### Model: `0180168492` (scenes: `"pp_west_side_10_angle_halfpi"`, `"pp_west_side_10_angle_pi"`)

#### Learning stage (90° range for initial random angles)
<table>
   <tr>
      <td align="center">demo 1: after 1k episodes</td>
      <td align="center">demo 2: after 2k episodes</td>
      <td align="center">demo 3: after 3k episodes</td>      
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/43b48ed5-8dad-4a2c-9a86-8305b44a58a9"><img src="https://github.com/pklesk/qlparking/assets/23095311/57151d88-75a7-4ce1-b791-5e54407460a4"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/020d03a7-3e2e-4b01-8e11-cfe4739825d2"><img src="https://github.com/pklesk/qlparking/assets/23095311/201b1d5b-30c6-4eb6-86f3-ad018d29c711"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/234c7443-8766-47ad-bd5f-9095d34dacaf"><img src="https://github.com/pklesk/qlparking/assets/23095311/b4e0d333-065a-48da-a551-248cebb828c7"/></a></td>
    </tr>
</table>

#### Testing stage 1 (90° range for initial random angles) - generalization after 10k episodes
<table>
   <tr>
      <td align="center">demo 4:</td>
      <td align="center">demo 5:</td>
      <td align="center">demo 6:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/7a4fb0bc-4368-4689-8135-3276233b352e"><img src="https://github.com/pklesk/qlparking/assets/23095311/eb56e614-f99a-4ad1-b8ec-76f270f2e02c"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/0ec3bf22-6dd0-42fb-b030-ee717318340a"><img src="https://github.com/pklesk/qlparking/assets/23095311/abaff328-c343-492c-8e0a-3d9b59b7ffde"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/cc0e47d6-775c-4f14-aec1-ff5a50280700"><img src="https://github.com/pklesk/qlparking/assets/23095311/f12eb69a-1bca-45c6-90bf-24f62cb6b20a"/></a></td>
    </tr>    
</table>

#### Testing stage 2 (180° range for initial random angles) - extrapolative generalization after 10k episodes
<table>
   <tr>
      <td align="center">demo 7:</td>
      <td align="center">demo 8:</td>
      <td align="center">demo 9:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/44c9f0b5-4a65-4449-b422-f84e39536865"><img src="https://github.com/pklesk/qlparking/assets/23095311/b5a045c1-31b2-4d1c-a78c-e3cfc54fe098"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/a0e0400c-1062-4f77-b7dc-5d269b94d2b2"><img src="https://github.com/pklesk/qlparking/assets/23095311/1546797b-a26d-4dd9-9e5f-ff45e561842f"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/1a8d0810-46b4-4134-915e-f01dcafd30e6"><img src="https://github.com/pklesk/qlparking/assets/23095311/08fbdfbe-2820-431d-8c1e-639ed2f5b1c0"/></a></td>
    </tr>

### Model: `4123751078` (scene: `"pp_middle_side_20_angle_twopi"`, 360° range for initial random angles)

#### Testing stage - generalization after 10k episodes

<table>
   <tr>
      <td align="center">demo 13:</td>
      <td align="center">demo 14:</td>
      <td align="center">demo 15:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/b65d6a0e-a4fb-4dda-a1dd-7d60a2a8fe86"><img src="https://github.com/pklesk/qlparking/assets/23095311/7e817c05-351a-4072-89f5-cd1e6be824b8"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/866305e6-2b66-4ed1-99c3-9b17a20c96ba"><img src="https://github.com/pklesk/qlparking/assets/23095311/11825a29-8438-4644-9bcb-7b7c87e607cd"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/fd24bf1d-842f-4ff2-aef7-8d6bb01500f7"><img src="https://github.com/pklesk/qlparking/assets/23095311/736a0441-58a8-4e5f-8b8d-0fb0162582ac"/></a></td>
    </tr>    
</table>

### Model: `4123751078` (scene: `"pp_random_car_random_side_20"`, 360° range also for initial random angles of park place, state representation switched from `"dv_flfrblbr2s_da"` to `"dv_flfrblbr2s_da_invariant`")

#### Testing stage - generalization after 10k episodes

<table>
   <tr>
      <td align="center">demo 16:</td>
      <td align="center">demo 17:</td>
      <td align="center">demo 18:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/f6389f6b-3dc2-46b0-ab72-1ba339972211"><img src="https://github.com/pklesk/qlparking/assets/23095311/3d6b7a62-743f-4386-998d-9d8a1497115b"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/a0cb9fec-12bc-47ed-bf9a-3877954c9293"><img src="https://github.com/pklesk/qlparking/assets/23095311/4aaa75b7-a904-48fb-9923-06bd912bf768"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/4adf873b-2716-49d1-bebf-4eb539b90c28"><img src="https://github.com/pklesk/qlparking/assets/23095311/50433dd0-97b0-45bd-98bf-4fb5e0c17c73"/></a></td>
    </tr>    
</table>

### First trials with obstacles and sensors - model: `0761736806` (scene: `"pp_middle_obstacles_oppdist_1_side_10_angle_pi"`, representation `"dv_flfrblbr2s_da_invariant_sensors`")

<table>
   <tr>
      <td align="center">demo 19:</td>
      <td align="center">demo 20:</td>
      <td align="center">demo 21:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/8cdb1b36-002c-47d4-a15e-4396c5e69bff"><img src="https://github.com/pklesk/qlparking/assets/23095311/b7f0a00b-233e-4dbd-a191-043df9e56c90"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/7fd077d5-27c9-413a-b703-c66478626462"><img src="https://github.com/pklesk/qlparking/assets/23095311/903f1879-c0f1-4774-9295-780d19d8ba57"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/39ac961e-5acb-4885-8903-1d764abceeca"><img src="https://github.com/pklesk/qlparking/assets/23095311/3d147961-91a2-4d1a-a4d5-e5ffae07e62f"/></a></td>
    </tr>    
</table>



