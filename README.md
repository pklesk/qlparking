# Double Q-Learning for a Simple Parking Problem

## Sample videos

### Model: `2354513149` (scenes: `"pp_west_side_10_angle_halfpi"`, `"pp_west_side_10_angle_pi"`)

#### Learning stage (90° range for initial random angles)
<table>
   <tr>
      <td align="center">demo 1: after 1k episodes</td>
      <td align="center">demo 2: after 2k episodes</td>
      <td align="center">demo 3: after 3k episodes</td>      
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/9487db7b-a427-47a2-afd7-62947f58ace8"><img src="https://github.com/pklesk/qlparking/assets/23095311/775a491d-ad7b-4e34-a140-4e9d678ef5d1"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/36250855-63cd-4f26-911e-b972efc6fe40"><img src="https://github.com/pklesk/qlparking/assets/23095311/e9a59e4c-e883-43f0-9ea6-da15f9b86d86"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/fdd1ee09-d866-4a6a-948f-3f18b7b3b2e1"><img src="https://github.com/pklesk/qlparking/assets/23095311/3064e348-947c-4440-bcc0-28efb847e92f"/></a></td>
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
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/1d886842-adc5-4273-ad3e-95f6dbce12ef"><img src="https://github.com/pklesk/qlparking/assets/23095311/92d23a80-eb75-473c-a9ec-acf28746bc17"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/bc1adb70-df15-474c-a75c-a479a6c408fd"><img src="https://github.com/pklesk/qlparking/assets/23095311/58636ed2-6e13-4282-874b-64afff9de649"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/dad27043-045c-42b6-ba22-ab7041be2088"><img src="https://github.com/pklesk/qlparking/assets/23095311/e351ea7e-71c9-446e-8602-743a95b6bce3"/></a></td>
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
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/91c950d7-5dfe-490a-8118-284621c34b4c"><img src="https://github.com/pklesk/qlparking/assets/23095311/ce913a11-1918-411c-8ad1-1dd2433fd59f"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/a0e0400c-1062-4f77-b7dc-5d269b94d2b2"><img src="https://github.com/pklesk/qlparking/assets/23095311/1546797b-a26d-4dd9-9e5f-ff45e561842f"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/1a8d0810-46b4-4134-915e-f01dcafd30e6"><img src="https://github.com/pklesk/qlparking/assets/23095311/08fbdfbe-2820-431d-8c1e-639ed2f5b1c0"/></a></td>
    </tr>
   <tr>
      <td align="center">demo 10:</td>
      <td align="center">demo 11:</td>
      <td align="center">demo 12:</td>
   </tr>   
   <tr>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/6dfd0204-acb2-4420-97ee-e079b664d2c1"><img src="https://github.com/pklesk/qlparking/assets/23095311/4c01d35b-ef49-4375-b2b6-941c5de988f2"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/50029e8c-327d-4652-8f82-b1bfbd559c5b"><img src="https://github.com/pklesk/qlparking/assets/23095311/00bf89b4-6115-4238-94e2-710fb804c8e6"/></a></td>
      <td><a href="https://github.com/pklesk/qlparking/assets/23095311/fcb896a3-131c-4d32-b14e-ef5f66fdbf0f"><img src="https://github.com/pklesk/qlparking/assets/23095311/f2dfb92b-ade9-41bb-a252-be71e3202697"></a></td>
    </tr>   
</table>

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



