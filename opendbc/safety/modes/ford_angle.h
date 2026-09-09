#pragma once
// Shared experimental Angle checks. Production permission is independently denied.
// This inverse is algebraic corroboration, NOT a validated EPS transfer model.
// Production Ford safety remains unchanged and rejects nonzero path angle.
// Firmware has no libm; all inputs below are bounded finite wire-derived values.

static uint32_t ford_a3_rx_time[3];
static bool ford_a3_rx_seen[3];
static bool ford_a3_rx_valid[3];
static uint32_t ford_a3_tx_time;
static bool ford_a3_tx_seen;
static int ford_a3_angle_last;
static int ford_a3_profile;
static unsigned int ford_a3_reasons;
static float ford_a3_equivalent_curvature;
static bool ford_a3_pending_valid;
static int ford_a3_pending_angle;
static int ford_a3_pending_desired;
static int ford_a3_pending_power;
static uint32_t ford_a3_pending_time;
static CurvatureSteeringState ford_a3_production_history;
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

static float ford_a3_abs(float x) { return x < 0.f ? -x : x; }
static int ford_a3_round(float x) {
  int whole = (int)x;
  float fraction = x - (float)whole;
  return whole + (fraction >= .5f ? 1 : (fraction <= -.5f ? -1 : 0));
}

static float ford_a3_interp(float x, float a, float b, float lo, float hi) {
  float f = (x-a)/(b-a);
  f = f < 0.f ? 0.f : (f > 1.f ? 1.f : f);
  return lo + f * (hi-lo);
}
static float ford_a3_gain(float curvature, float speed) {
  const float profiles[4][4] = {{1.425f,1.f,1.f,1.f}, {.95f,1.f,1.f,1.f}, {1.425f,.9f,1.f,1.f}, {1.425f,1.f,.9f,.9f}};
  const float *p = profiles[ford_a3_profile];
  float low = ford_a3_interp(speed, 13.5f, 26.82f, 1.f, p[0]*p[3]);
  float high = ford_a3_interp(speed, 13.5f, 26.82f, 1.3f*p[1], p[0]*p[2]);
  return ford_a3_interp(ford_a3_abs(curvature), .0007f, .001f, low, high);
}
static float ford_a3_inverse(float angle, float speed) {
  float lo = 0.f, hi = 1.f;
  for (int i=0; i<32; i++) {
    float mid = (lo+hi)*.5f;
    if (mid*speed*ford_a3_gain(mid,speed) < ford_a3_abs(angle)) lo=mid; else hi=mid;
  }
  return angle < 0.f ? -(lo+hi)*.5f : (lo+hi)*.5f;
}
static void ford_a3_reset(void) {
  ford_a3_pending_valid=false;
  ford_a3_production_history = (CurvatureSteeringState){0};
  for (int i=0; i<3; i++) { ford_a3_rx_time[i]=0; ford_a3_rx_seen[i]=false; ford_a3_rx_valid[i]=false; }
  ford_a3_tx_time=0; ford_a3_tx_seen=false;
  ford_a3_angle_last=0; ford_a3_profile=0; ford_a3_reasons=0; ford_a3_equivalent_curvature=0.f;
}
static void ford_a3_observe_rx(const CANPacket_t *p, bool accepted) {
  int i = p->addr == FORD_BrakeSysFeatures ? 0 : p->addr == FORD_EngVehicleSpThrottle2 ? 1 : p->addr == FORD_Yaw_Data_FD1 ? 2 : -1;
  if (i>=0 && p->bus==0) { ford_a3_rx_seen[i]=true; ford_a3_rx_valid[i]=accepted && GET_LEN(p)==8; if (ford_a3_rx_valid[i]) ford_a3_rx_time[i]=microsecond_timer_get(); }
}
static bool ford_a3_check(const CANPacket_t *p) {
  ford_a3_pending_valid=false;
  ford_a3_reasons=0;
  if (relay_malfunction) ford_a3_reasons|=A3_RELAY;
  unsigned int mode=(p->data[0]>>4)&7U;
  int c=((p->data[2]<<3)|(p->data[3]>>5))-1000;
  int a=(((p->data[3]&31)<<6)|(p->data[4]>>2))-1000;
  int o=(((p->data[4]&3)<<8)|p->data[5])-512;
  int r=((p->data[6]<<3)|(p->data[7]>>5))-1024;
  bool active=mode!=0;
  if (p->bus!=0 || GET_LEN(p)!=8 || mode>1 || c!=0 || o!=0 || r!=0) ford_a3_reasons|=A3_MODE;
  if (ford_a3_profile<0 || ford_a3_profile>3) ford_a3_reasons|=A3_PROFILE;
  if (!active && a!=0) ford_a3_reasons|=A3_INACTIVE;
  if (a < -1000 || a > 1047) ford_a3_reasons|=A3_WIRE_RANGE;
  uint32_t now=microsecond_timer_get();
  float speed=vehicle_speed.values[0]/VEHICLE_SPEED_FACTOR;
  if (active) {
    for (int i=0;i<3;i++) if (!ford_a3_rx_seen[i] || !ford_a3_rx_valid[i] || (now-ford_a3_rx_time[i])>100000U) ford_a3_reasons|=A3_FRESHNESS;
    if (speed<1.f || speed>60.f) ford_a3_reasons|=A3_SPEED;
    float path_rate = speed<=15.f ? ford_a3_interp(speed,10.f,15.f,.055f,.0425f) : ford_a3_interp(speed,15.f,25.f,.0425f,.009f);
    if (ford_a3_abs((a-ford_a3_angle_last)*.0005f)>path_rate+1e-7f) ford_a3_reasons|=A3_PATH_RATE;
  }
  // An inactive frame clears no safety violation and grants no special reentry.
  if (ford_a3_reasons) return false;
  ford_a3_equivalent_curvature=active ? ford_a3_inverse(a*.0005f,speed) : 0.f;
  // The donor callsite negates host path angle before encoding. Inverting the
  // final wire angle therefore yields the same sign as trusted Ford yaw curvature.
  int equivalent_can=ford_a3_round(ford_a3_equivalent_curvature*50000.f);
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
  if (steer_curvature_cmd_checks(equivalent_can,0,active,FORD_STEERING_LIMITS)) ford_a3_reasons|=A3_STOCK_ENVELOPE;
  // speed_mismatch_check can revoke permission before the stock rate helper.
  // That helper always increments or rolls its buckets when executed, so
  // changed bucket state proves invocation; unchanged state is not attributed.
  bool aggregate_checked = curvature_state.rt_msgs != rt_current_before ||
    curvature_state.rt_msgs_prev != rt_previous_before || curvature_state.ts_check_last != rt_roll_before;
  if (aggregate_checked && aggregate_violation) ford_a3_reasons|=A3_CADENCE;
  ford_a3_pending_desired = curvature_state.desired_last;
  ford_a3_pending_power = curvature_state.steer_power_last;
  curvature_state.desired_last = accepted_desired;
  curvature_state.steer_power_last = accepted_power;
  ford_a3_pending_valid = ford_a3_reasons == 0U;
  ford_a3_pending_angle = a;
  ford_a3_pending_time = now;
  return ford_a3_pending_valid;
}

// Caller commits only after final admission. Never called by production firmware.
static inline void ford_a3_commit(void) {
  if (ford_a3_pending_valid) {
    curvature_state.desired_last = ford_a3_pending_desired;
    curvature_state.steer_power_last = ford_a3_pending_power;
    ford_a3_angle_last = ford_a3_pending_angle;
    ford_a3_tx_time = ford_a3_pending_time;
    ford_a3_tx_seen = true;
    ford_a3_pending_valid = false;
  }
}

static bool ford_a3_production_tx(const CANPacket_t *msg) {
  // Independent attempt buckets: experimental probes must not alter A2 history.
  CurvatureSteeringState a2_history = curvature_state;
  curvature_state = ford_a3_production_history;
  curvature_state.meas = a2_history.meas;
  (void)ford_a3_check(msg);
  ford_a3_production_history = curvature_state;
  curvature_state = a2_history;
  ford_a3_pending_valid = false;
  // No profile, safetyParam, signing mode or host claim can override this block.
  return false;
}
