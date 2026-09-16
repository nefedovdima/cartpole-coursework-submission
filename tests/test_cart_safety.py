"""Independent physical oracles for the optional filter; stdlib only, no Drake."""
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
import math
import unittest

from cartpole.safety import SafetyLimits, assess_cart_state, filter_cart_command


def oracle(x, v, u, h, limits):
    """High-precision physics + full braking to rest after the held command.

    No analytic action-bound/root implementation is shared with the filter.
    Evaluate the polynomial at its derivative's zero and at both endpoints;
    extend by maximum braking to a rest point to check future viability.
    """
    with localcontext() as c:
        # Enough to retain the exact binary-float polynomial at a zero-margin
        # full-braking boundary; 70 digits rounded an exact equality outward.
        c.prec=180
        x,v,u,h,b,w,a=map(Decimal.from_float,map(float,(x,v,u,h,limits.inner_position,
                                        limits.inner_velocity,limits.acceleration_limit)))
        end_x=x+v*h+u*h*h/2;end_v=v+u*h
        values=[x,end_x]
        if u:
            t=-v/u
            if 0<t<h:values.append(x+v*t+u*t*t/2)
        if end_v:
            brake=-a if end_v>0 else a
            brake_time=-end_v/brake
            stop=end_x+end_v*brake_time+brake*brake_time*brake_time/2
            values.append(stop)
        return abs(u)<=a and min(values)>=-b and max(values)<=b and max(abs(v),abs(end_v))<=w


class CartSafetyTests(unittest.TestCase):
    def setUp(self):
        self.limits=SafetyLimits()

    def choose(self,x,v,u,**kwargs):
        return filter_cart_command(x=x,v=v,proposed_u=u,**kwargs)

    def assert_safe(self,x,v,result,h=.01,limits=None):
        self.assertTrue(result.feasible,result)
        self.assertTrue(oracle(x,v,result.u_filtered,h,limits or self.limits),result)
        self.assertGreaterEqual(result.prediction.position_margin_interval,0)
        self.assertGreaterEqual(result.prediction.terminal.right_stopping,0)
        self.assertGreaterEqual(result.prediction.terminal.left_stopping,0)

    def test_safe_commands_are_bitwise_unchanged(self):
        for u in (0.,-0.,1.125,-4.,4.,math.nextafter(0.,1.)):
            r=self.choose(.01,.01,u)
            self.assertFalse(r.intervened);self.assertEqual(r.u_filtered.hex(),u.hex())
            self.assertEqual(r.correction_abs,0)

    def test_near_either_edge_outward_command_is_restricted(self):
        for side in (-1,1):
            x,v,u=side*.234,side*.2,side*4
            r=self.choose(x,v,u)
            self.assert_safe(x,v,r)
            self.assertTrue(r.intervened)
            self.assertLess(side*r.u_filtered,4)
            self.assertIn('future_'+('right' if side>0 else 'left')+'_stopping',r.proposal_violations)

    def test_motion_away_from_edge_is_not_blindly_braked(self):
        for side in (-1,1):
            x,v=side*.239,-side*.05
            r=self.choose(x,v,0.)
            self.assert_safe(x,v,r);self.assertFalse(r.intervened)

    def test_rest_on_inner_boundary_zero_and_inward_commands(self):
        for side in (-1,1):
            x=side*self.limits.inner_position
            for u in (0.,-side*1.):
                r=self.choose(x,0,u);self.assert_safe(x,0,r)
                self.assertEqual(r.u_filtered,u)
            r=self.choose(x,0,side*4)
            self.assert_safe(x,0,r);self.assertLessEqual(side*r.u_filtered,0)

    def test_interior_excursion_with_both_samples_safe(self):
        b=self.limits.inner_position;x=b-2e-5;v=.01;u=-2.;h=.01
        end=x+v*h+.5*u*h*h
        peak=x+.01*.005+.5*u*.005**2
        self.assertLessEqual(end,b);self.assertGreater(peak,b)
        self.assertTrue(assess_cart_state(x=x,v=v).admissible)
        r=self.choose(x,v,u)
        self.assertIn('position_right_interval',r.proposal_violations)
        self.assertLessEqual(r.u_filtered,-2.5)
        self.assert_safe(x,v,r)

    def test_braking_and_reversal_inside_step_can_be_safe(self):
        r=self.choose(.23,.01,-4.)
        self.assert_safe(.23,.01,r);self.assertFalse(r.intervened)
        self.assertLess(r.prediction.v_end,0)
        self.assertGreater(r.prediction.x_max,max(.23,r.prediction.x_end))

    def test_exact_tangency_returns_inward_without_false_failure(self):
        # Binary-exact contact: t_turn=1/256, x_peak=1/8. End is inside.
        limits=SafetyLimits(position_limit=.125,position_reserve=0,intervention_inset=0)
        v=1/64;u=-4.;x=.125-v*v/8
        r=self.choose(x,v,u,limits=limits)
        self.assert_safe(x,v,r,limits=limits)
        self.assertEqual(r.prediction.x_max,.125);self.assertFalse(r.intervened)

    def test_endpoint_inside_but_future_stop_outside_is_rejected(self):
        r=self.choose(.234,.2,4.)
        self.assertLess(.234+.2*.01+2*.01**2,.24)
        self.assertGreater(.234+.2*.01+2*.01**2+.24**2/8,.24)
        self.assertTrue(r.intervened);self.assert_safe(.234,.2,r)

    def test_acceleration_saturation_remains_closest_at_center(self):
        for u in (-1e308,-9.,9.,1e308):
            r=self.choose(0,0,u)
            self.assertEqual(r.u_filtered,math.copysign(4,u))
            self.assertEqual(r.numerical_inset,0)
            self.assert_safe(0,0,r)

    def test_velocity_limit_for_both_directions(self):
        # With .24/4 the stopping envelope already implies |v|<sqrt(3.84)<2.
        # A smaller diagnostic velocity limit makes the speed constraint active.
        limits=replace(self.limits,velocity_limit=.2,velocity_reserve=0,intervention_inset=0)
        for side in (-1,1):
            r=self.choose(0,side*.19,side*4,limits=limits)
            self.assert_safe(0,side*.19,r,limits=limits)
            self.assertAlmostEqual(r.u_filtered,side*1.,places=13)
            self.assertLessEqual(r.prediction.max_abs_velocity,.2)
        self.assertFalse(self.choose(0,2.0000001,0).feasible)

    def test_boundary_of_viability_requires_full_braking(self):
        limits=SafetyLimits(position_limit=.125,position_reserve=0,intervention_inset=0)
        x=.125-.25**2/8
        for side in (-1,1):
            r=self.choose(side*x,side*.25,side*4,limits=limits)
            self.assert_safe(side*x,side*.25,r,limits=limits)
            self.assertEqual(r.u_filtered,-side*4.)
            self.assertEqual(r.feasible_interval,(-side*4.,-side*4.))

    def test_adjacent_floats_around_viability_are_not_tolerance_clipped(self):
        limits=SafetyLimits(position_limit=.125,position_reserve=0)
        boundary=.125-.25**2/8
        inside=math.nextafter(boundary,-math.inf)
        outside=math.nextafter(boundary,math.inf)
        self.assertTrue(self.choose(inside,.25,4,limits=limits).feasible)
        r=self.choose(outside,.25,-4,limits=limits)
        self.assertFalse(r.feasible);self.assertIsNone(r.u_filtered)
        self.assertIn('right_stopping_distance',r.state.reasons)

    def test_physically_impossible_start_has_no_fake_emergency_action(self):
        for side in (-1,1):
            r=self.choose(side*.23,side*.5,-side*4)
            self.assertFalse(r.state.viable_at_requested_limits)
            self.assertFalse(r.feasible);self.assertIsNone(r.u_filtered)
            self.assertIsNone(r.intervened);self.assertIsNone(r.prediction)
            self.assertAlmostEqual(min(r.state.requested.right_stopping,r.state.requested.left_stopping),-.02125)

    def test_inner_reserve_rejection_is_distinguished_from_physical_impossibility(self):
        r=self.choose(.24-.5e-6,0,0)
        self.assertFalse(r.state.admissible)
        self.assertTrue(r.state.viable_at_requested_limits)
        self.assertIsNone(r.u_filtered)

    def test_nonfinite_inputs_and_non_scalar_types(self):
        for name in ('x','v','proposed_u','dt'):
            for value in (math.nan,math.inf,-math.inf):
                kw=dict(x=0.,v=0.,proposed_u=0.,dt=.01);kw[name]=value
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):filter_cart_command(**kw)
            for value in (True,'0',None,[0],(0,),complex(0)):
                kw=dict(x=0.,v=0.,proposed_u=0.,dt=.01);kw[name]=value
                with self.subTest(name=name,value=value),self.assertRaises(TypeError):filter_cart_command(**kw)

    def test_dt_is_positive_bounded_and_validated_even_for_bad_start(self):
        for dt in (0,-.01,1e-12,.010000001,1e308):
            with self.assertRaises(ValueError):self.choose(.5,3,0,dt=dt)
        for dt in (1e-9,.003,.01):
            r=self.choose(0,0,4,dt=dt);self.assert_safe(0,0,r,h=dt)
        delta=100*.01-99*.01  # slightly above .01 in binary arithmetic
        r=self.choose(0,0,4,dt=delta)
        self.assert_safe(0,0,r,h=delta)
        self.assertEqual(r.prediction.v_end,4*delta)
        limits=SafetyLimits(position_limit=.125,position_reserve=0,max_dt=.25)
        with self.assertRaisesRegex(ValueError,'proved sampled'):
            self.choose(0,0,0,dt=math.nextafter(.25,math.inf),limits=limits)

    def test_sampled_data_assumptions_are_enforced(self):
        for changes in (dict(acceleration_limit=0),dict(position_reserve=.24),
                        dict(velocity_reserve=2),dict(min_dt=0),dict(max_dt=1),
                        dict(max_dt=.02,velocity_limit=.05),dict(position_reserve=1e-30)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):SafetyLimits(**changes)

    def test_symmetry_on_deterministic_grid(self):
        for x in (-.23,-.1,0,.1,.23):
            for v in (-.2,0,.2):
                for u in (-9.,-1.,0.,1.,9.):
                    first=self.choose(x,v,u);second=self.choose(-x,-v,-u)
                    self.assertEqual(first.feasible,second.feasible)
                    if first.feasible:
                        self.assertAlmostEqual(first.u_filtered,-second.u_filtered,places=13)
                        self.assert_safe(x,v,first)

    def test_projection_agrees_with_independent_feasible_action_grid(self):
        limits=replace(self.limits,intervention_inset=0)
        cases=[(.234,.2,4),(.239979,.01,-2),(-.234,-.2,-4),(.20,-.3,8)]
        for x,v,proposed in cases:
            r=self.choose(x,v,proposed,limits=limits);self.assert_safe(x,v,r,limits=limits)
            grid=[-4+8*i/800 for i in range(801)]
            feasible=[u for u in grid if oracle(x,v,u,.01,limits)]
            self.assertTrue(feasible)
            self.assertLessEqual(abs(r.u_filtered-proposed),min(abs(u-proposed) for u in feasible)+1e-12)
            toward=math.copysign(1e-7,proposed-r.u_filtered)
            self.assertFalse(oracle(x,v,r.u_filtered+toward,.01,limits))

    def test_repeated_adversarial_commands_without_state_projection(self):
        for mode in ('right','left','alternating'):
            x=v=0.
            for k in range(1200):
                u=100. if mode=='right' else -100. if mode=='left' else (100. if (k//150)%2==0 else -100.)
                r=self.choose(x,v,u)
                self.assert_safe(x,v,r)
                # Independent floating-point physical recurrence, never clamp state.
                x=math.fsum((x,v*.01,.5*r.u_filtered*.01**2))
                v=math.fsum((v,r.u_filtered*.01))
                self.assertLessEqual(abs(x),.24)
                self.assertLessEqual(abs(v),2.)

    def test_near_zero_velocity_and_boundary_input(self):
        for v in (1e-300,-1e-300,math.nextafter(0.,1.),-math.nextafter(0.,1.)):
            r=self.choose(0,v,0)
            self.assertTrue(r.feasible);self.assertEqual(r.u_filtered,0)
        r=self.choose(self.limits.inner_position,1e-300,-4)
        self.assertFalse(r.state.admissible)
        self.assertTrue(r.state.viable_at_requested_limits)

    def test_nonempty_action_on_kernel_grid_up_to_proved_dt_bound(self):
        limits=SafetyLimits(position_limit=.125,position_reserve=0,max_dt=.25)
        # max_dt is exactly sqrt(2*L/A). States and their stop points are
        # binary-exact; this diagnoses the ZOH proof, not a changed experiment.
        for x in (-.125,-.0625,0,.0625,.125):
            for v in (-1.,-.5,-.125,0.,.125,.5,1.):
                stop=x+math.copysign(v*v/8,v)
                if not -.125<=stop<=.125:
                    continue
                for h in (.0078125,.125,.25):
                    for u in (-4.,4.):
                        r=self.choose(x,v,u,dt=h,limits=limits)
                        with self.subTest(x=x,v=v,h=h,u=u):
                            self.assert_safe(x,v,r,h=h,limits=limits)

    def test_uncertifiable_numeric_bounds_do_not_create_an_emergency_command(self):
        from unittest.mock import patch
        # Fault injection into the root calculation, not a surrogate simulator.
        with patch('cartpole.safety._interval',return_value=(4.,-4.,True,True)):
            r=self.choose(.234,.2,4.)
            self.assertTrue(r.state.admissible)
            self.assertFalse(r.feasible);self.assertIsNone(r.u_filtered)
            self.assertEqual(r.reason,'no_certified_float_action')
            self.assertEqual(self.choose(0,0,1.).u_filtered,1.)

    def test_diagnostics_are_immutable_json_finite_and_do_not_claim_execution(self):
        import json
        r=self.choose(.234,.2,4)
        json.dumps(asdict(r),allow_nan=False)
        self.assertEqual(r.correction,r.u_filtered-r.u_proposed)
        self.assertGreaterEqual(r.prediction.position_margin_interval,0)
        self.assertNotIn('applied_acceleration',asdict(r))
        with self.assertRaises(AttributeError):r.u_filtered=0


if __name__=='__main__':
    unittest.main()
