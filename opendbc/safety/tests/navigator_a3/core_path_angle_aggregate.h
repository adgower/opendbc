#pragma once
// HOST-ONLY aggregate-timing hypothesis evaluator. Not included by safety.h or firmware.
// This inverse is algebraic corroboration, NOT a validated EPS transfer model.
// Production Ford safety remains unchanged and rejects nonzero path angle.
#include <math.h>

static uint32_t a3_rx_time[3];
static bool a3_rx_seen[3];
static bool a3_rx_valid[3];
static uint32_t a3_tx_time;
static bool a3_tx_seen;
static int a3_angle_last;
static int a3_profile;
static unsigned int a3_reasons;
static float a3_equivalent_curvature;
// Rejection bits are additional to traced stock curvature predicates.
#define A3_MODE 1U
#define A3_FRESHNESS 2U
#define A3_CADENCE 4U
#define A3_PROFILE 8U
#define A3_INACTIVE 16U
#define A3_STOCK_ENVELOPE 32U
#define A3_SPEED 64U
#define A3_WIRE_RANGE 128U
#define A3_RELAY 256U
#define A3_PATH_RATE 512U

static float a3_interp(float x, float a, float b, float lo, float hi) {
  float f = fminf(fmaxf((x-a)/(b-a), 0.f), 1.f);
  return lo + f * (hi-lo);
}
static float a3_gain(float curvature, float speed) {
  const float profiles[4][4] = {{1.425f,1.f,1.f,1.f}, {.95f,1.f,1.f,1.f}, {1.425f,.9f,1.f,1.f}, {1.425f,1.f,.9f,.9f}};
  const float *p = profiles[a3_profile];
  float low = a3_interp(speed, 13.5f, 26.82f, 1.f, p[0]*p[3]);
  float high = a3_interp(speed, 13.5f, 26.82f, 1.3f*p[1], p[0]*p[2]);
  return a3_interp(fabsf(curvature), .0007f, .001f, low, high);
}
static float a3_inverse(float angle, float speed) {
  float lo = 0.f, hi = 1.f;
  for (int i=0; i<32; i++) {
    float mid = (lo+hi)*.5f;
    if (mid*speed*a3_gain(mid,speed) < fabsf(angle)) lo=mid; else hi=mid;
  }
  return copysignf((lo+hi)*.5f, angle);
}
static void a3_reset(void) {
  memset(a3_rx_time, 0, sizeof(a3_rx_time)); memset(a3_rx_seen, 0, sizeof(a3_rx_seen));
  memset(a3_rx_valid, 0, sizeof(a3_rx_valid)); a3_tx_time=0; a3_tx_seen=false;
  a3_angle_last=0; a3_profile=0; a3_reasons=0; a3_equivalent_curvature=0.f;
}
static void a3_observe_rx(const CANPacket_t *p, bool accepted) {
  int i = p->addr == FORD_BrakeSysFeatures ? 0 : p->addr == FORD_EngVehicleSpThrottle2 ? 1 : p->addr == FORD_Yaw_Data_FD1 ? 2 : -1;
  if (i>=0 && p->bus==0) { a3_rx_seen[i]=true; a3_rx_valid[i]=accepted && GET_LEN(p)==8; if (a3_rx_valid[i]) a3_rx_time[i]=microsecond_timer_get(); }
}
static bool a3_admit(const CANPacket_t *p) {
  a3_reasons=0;
  if (relay_malfunction) a3_reasons|=A3_RELAY;
  unsigned int mode=(p->data[0]>>4)&7U;
  int c=((p->data[2]<<3)|(p->data[3]>>5))-1000;
  int a=(((p->data[3]&31)<<6)|(p->data[4]>>2))-1000;
  int o=(((p->data[4]&3)<<8)|p->data[5])-512;
  int r=((p->data[6]<<3)|(p->data[7]>>5))-1024;
  bool active=mode!=0;
  if (p->bus!=0 || GET_LEN(p)!=8 || mode>1 || c!=0 || o!=0 || r!=0) a3_reasons|=A3_MODE;
  if (a3_profile<0 || a3_profile>3) a3_reasons|=A3_PROFILE;
  if (!active && a!=0) a3_reasons|=A3_INACTIVE;
  if (a < -1000 || a > 1047) a3_reasons|=A3_WIRE_RANGE;
  uint32_t now=microsecond_timer_get();
  float speed=vehicle_speed.values[0]/VEHICLE_SPEED_FACTOR;
  if (active) {
    for (int i=0;i<3;i++) if (!a3_rx_seen[i] || !a3_rx_valid[i] || (now-a3_rx_time[i])>100000U) a3_reasons|=A3_FRESHNESS;
    if (speed<1.f || speed>60.f) a3_reasons|=A3_SPEED;
    float path_rate = speed<=15.f ? a3_interp(speed,10.f,15.f,.055f,.0425f) : a3_interp(speed,15.f,25.f,.0425f,.009f);
    if (fabsf((a-a3_angle_last)*.0005f)>path_rate+1e-7f) a3_reasons|=A3_PATH_RATE;
  }
  // An inactive frame clears no safety violation and grants no special reentry.
  if (a3_reasons) return false;
  a3_equivalent_curvature=active ? a3_inverse(a*.0005f,speed) : 0.f;
  // The donor callsite negates host path angle before encoding. Inverting the
  // final wire angle therefore yields the same sign as trusted Ford yaw curvature.
  int equivalent_can=(int)roundf(a3_equivalent_curvature*50000.f);
  // The frozen stock helper is byte-identical to sunnypilot/opendbc
  // f95f996f5917dcbbf2e32fe51b606a24cf836af6 safety/lateral.h:177-196.
  // It checks current+previous count BEFORE increment/125 ms roll, with
  // MAX_RT_INTERVAL=250000 us and FORD_STEERING_LIMITS.frequency=20.
  // Keep attempt-rate buckets, but commit desired history only on acceptance.
  int accepted_desired = curvature_state.desired_last;
  int accepted_power = curvature_state.steer_power_last;
  int max_rt_msgs = ((float)FORD_STEERING_LIMITS.frequency * MAX_RT_INTERVAL / 1e6 * 1.2) + 1;
  uint32_t rt_current_before = curvature_state.rt_msgs;
  uint32_t rt_previous_before = curvature_state.rt_msgs_prev;
  uint32_t rt_roll_before = curvature_state.ts_check_last;
  bool aggregate_violation = (int)(rt_current_before + rt_previous_before) > max_rt_msgs;
  if (steer_curvature_cmd_checks(equivalent_can,0,active,FORD_STEERING_LIMITS)) a3_reasons|=A3_STOCK_ENVELOPE;
  // speed_mismatch_check can revoke permission before the stock rate helper.
  // That helper always increments or rolls its buckets when executed, so
  // changed bucket state proves invocation; unchanged state is not attributed.
  bool aggregate_checked = curvature_state.rt_msgs != rt_current_before ||
    curvature_state.rt_msgs_prev != rt_previous_before || curvature_state.ts_check_last != rt_roll_before;
  if (aggregate_checked && aggregate_violation) a3_reasons|=A3_CADENCE;
  if (a3_reasons) {
    curvature_state.desired_last = accepted_desired;
    curvature_state.steer_power_last = accepted_power;
  }
  if (!a3_reasons) { a3_tx_time=now; a3_tx_seen=true; a3_angle_last=a; }
  return a3_reasons==0;
}
