from opendbc.car.ford.longitudinal_coordination import Torque
from opendbc.car.ford.longitudinal_progress_historical import HistoricalProgress

def seeded(points,now=1.1):
 p=HistoricalProgress();p.episode=0.;p.samples=[(0.,328.)]+points
 return p.expired(now,0.,Torque(*points[-1]))

def test_historical_witness_with_recent_progress_continues():
 points=[(.6+i*.02,156.-152.*i*.02) for i in range(26)]
 assert not seeded(points)

def test_rebound_return_is_not_new_low():
 points=[(.6+i*.02,80.) for i in range(26)]
 points=[(t,72. if t==.6 else 80. if t<.9 else 72.) for t,v in points]
 assert seeded(points)

def test_history_does_not_hide_recent_plateau():
 points=[(.6+i*.02,160.-160.*min(i*.02,.25)) for i in range(26)]
 assert seeded(points)

def test_historical_hypothesis_differs_from_current_rate():
 points=[(.6+i*.02,160.-(64.*(i*.02)/.26 if i*.02<=.26 else 64.+8.*(i*.02-.26)/.24)) for i in range(26)]
 assert not seeded(points)

import pytest,math
from opendbc.car.ford.longitudinal_coordination import BrakeReleaseProgress

@pytest.mark.parametrize('peak',[40.,100.,200.])
@pytest.mark.parametrize('rate',[0.,20.,100.])
def test_floor_domain_exact_original(peak,rate):
 a,b=BrakeReleaseProgress(),HistoricalProgress()
 for i in range(101):
  t=i*.02;y=max(8.,peak-rate*t)
  assert a.expired(t,0.,Torque(t,y))==b.expired(t,0.,Torque(t,y))
 assert a.samples==b.samples

@pytest.mark.parametrize('cut',[1.,1.08,1.2,1.48])
@pytest.mark.parametrize('pattern',['flat','alternate','return'])
def test_no_new_low_expires_within_quarter_second_plus_callback(cut,pattern):
 p=HistoricalProgress();lowest=float('inf');last_new_low=None;first_after=None
 for i in range(101):
  t=round(i*.02,9);base=1200. if t<=.1 else max(80.,1200.-1100.*(t-.1))
  if t<=cut:y=4*math.floor(base/4);plateau=y
  else:y=plateau+(0 if pattern=='flat' else 4*(i%2) if pattern=='alternate' else 8*(i%3==0))
  if y<lowest:lowest=y;last_new_low=t
  expired=p.expired(t,0.,Torque(t,y))
  if t>=1. and expired and first_after is None:first_after=t
  if t>cut+.27:break
 assert first_after is not None
 assert first_after<=max(1.,last_new_low+.27)+1e-9

@pytest.mark.parametrize('bad_span',[.38,.3])
def test_insufficient_historical_span_rejects(bad_span):
 points=[(1.1-bad_span+i*.02,160.-i*4.) for i in range(round(bad_span/.02)+1)]
 assert seeded(points)

def test_recycled_low_counterexample_requires_new_low_guard():
 # Strong historical fall, then permitted 8Nm rebound and return to old72Nm.
 # Recent floor and historical bounds alone both pass although last newlow is.7s.
 points=[]
 for i in range(26):
  t=round(.6+i*.02,9)
  y=160.-880.*(t-.6) if t<=.7 else 80. if t<1. else 72.
  points.append((t,round(y,6)))
 recent=[(t,y) for t,y in points if t>=.85]
 assert recent[0][1]-recent[-1][1]>=8.
 assert points[0][1]-points[-1][1]-8. >= .04*(328.+4.)*.5/.25
 assert seeded(points)
