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
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/fbd8124a-9bd0-4b9b-9450-aef40d7c1384"><img src="https://github.com/pklesk/qlparking/assets/23095311/072ba063-f97c-4bc6-b06b-3feb8d2d9e35"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/200067c3-4dae-402e-b0ef-66d259979668"><img src="https://github.com/pklesk/qlparking/assets/23095311/bcb72343-55da-43be-9630-425546461d76"/></a></td>
    </tr>
</table>   

### Model: `0623865367` (scene: `"pp_middle_side_20_angle_twopi"`, 360° range for initial random angles)

#### Testing stage - generalization after 10k episodes

<table>
   <tr>
      <td align="center">demo 10:</td>
      <td align="center">demo 11:</td>
      <td align="center">demo 12:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/dafbeaa7-0993-4ade-a761-d8452a615987"><img src="https://github.com/pklesk/qlparking/assets/23095311/e03feb58-8436-431e-9f34-24a861a49111"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/fe9171a2-3ea3-4dc3-9714-a83c5aa7c5ce"><img src="https://github.com/pklesk/qlparking/assets/23095311/d2241d03-9de5-4c9a-9438-07d75c49e79e"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/c48748e4-fee8-4452-a938-619e44d3de28"><img src="https://github.com/pklesk/qlparking/assets/23095311/b2150e4a-c2b0-4250-98bd-e63fe244ef99"/></a></td>
    </tr>    
</table>

### Model: `0623865367` (scene: `"pp_random_car_random_side_20"`, 360° range also for initial random angles of park place, state representation switched from `"dv_flfrblbr2s_dag"` to `"dv_flfrblbr2s_dag_invariant`")

#### Testing stage - generalization after 10k episodes

<table>
   <tr>
      <td align="center">demo 13:</td>
      <td align="center">demo 14:</td>
      <td align="center">demo 15:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/f7ee3714-729d-492a-b14c-7ee585e6c17f"><img src="https://github.com/pklesk/qlparking/assets/23095311/a27fe369-e30f-4ed6-a74c-8e12efaedea3"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/2d4ae11e-4ea7-44a9-b7dc-c6faa89336eb"><img src="https://github.com/pklesk/qlparking/assets/23095311/54065428-ada1-455d-a092-c904fb0955b9"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/8f92564c-db8e-4791-a0ea-ed272c7f0019"><img src="https://github.com/pklesk/qlparking/assets/23095311/ac0d3e1d-2e86-4770-b793-ce7623ad9642"/></a></td>
    </tr>    
</table>

### First trials with obstacles and sensors - model: `2914586007` (scene: `"pp_middle_obstacles_oppdist_1_side_10_angle_pi"`, representation `"dv_flfrblbr2s_dag_invariant_sensors`")

<table>
   <tr>
      <td align="center">demo 16:</td>
      <td align="center">demo 17:</td>
      <td align="center">demo 21:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/0016bc8d-9378-4c8e-a85a-3b4674a8ea5e"><img src="https://github.com/pklesk/qlparking/assets/23095311/3cdb4250-e63a-48d0-9f0e-41961974359b"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/f8293df4-061e-47f3-b54a-2483f59e10a6"><img src="https://github.com/pklesk/qlparking/assets/23095311/089cafa1-dfe4-4bab-a55c-b531f523cf7e"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/39ac961e-5acb-4885-8903-1d764abceeca"><img src="https://github.com/pklesk/qlparking/assets/23095311/3d147961-91a2-4d1a-a4d5-e5ffae07e62f"/></a></td>
    </tr>    
</table>
