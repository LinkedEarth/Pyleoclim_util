!---------------------------------------------------------------------------------
! @author: fzhu (fengzhu@usc.edu)
! 2017-06-17 13:30:38
!---------------------------------------------------------------------------------

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

module f2py_wwz
    implicit none

    contains

    subroutine wwa(taus, omegas, c, Neff, ts, pd_ys, nthread, nts, nt, nf, Neffs, a0, a1, a2)
        implicit none
        !---------------------------------------------------------------------------------
        integer, intent(in) :: Neff, nts, nt, nf, nthread
        double precision, intent(in) :: c
        double precision, dimension(nt), intent(in) :: taus
        double precision, dimension(nf), intent(in) :: omegas
        double precision, dimension(nts), intent(in) :: ts, pd_ys
        double precision, dimension(nt, nf), intent(out) :: Neffs, a0, a1, a2
        ! double precision, intent(out) :: t_total
        !---------------------------------------------------------------------------------
        integer :: k, j
        ! integer :: t_start, t_end, rate
        ! real :: t_lapse
        !---------------------------------------------------------------------------------
        ! t_total = 0.

        CALL OMP_SET_NUM_THREADS(nthread)

        !$OMP PARALLEL DO
        do k = 1, nf
            do j = 1, nt
                ! call system_clock(t_start, rate)
                call wwa_1g(taus(j), omegas(k), c, Neff, ts, pd_ys, Neffs(j, k), a0(j, k), a1(j, k), a2(j, k), nts)
                ! call system_clock(t_end)
                ! t_lapse = real(t_end - t_start) / real(rate)
                ! t_total = t_total + t_lapse
            end do
        end do
        !$OMP END PARALLEL DO

    end subroutine wwa

    subroutine wwa_1g(tau, omega, c, Neff, ts, pd_ys, Neff_loc, a0, a1, a2, nts)
        implicit none
        !---------------------------------------------------------------------------------
        integer, intent(in) :: nts, Neff
        double precision, intent(in) :: tau, omega, c
        double precision, dimension(nts), intent(in) :: ts, pd_ys
        double precision, intent(out) :: Neff_loc, a0, a1, a2
        !---------------------------------------------------------------------------------
        double precision, dimension(nts) :: dz, weights
        double precision :: sum_w
        double precision, dimension(nts) :: sin_basis, cos_basis, one_v
        double precision :: sin_one, cos_one, sin_cos, sin_sin, cos_cos
        double precision :: numerator, denominator, time_shift
        double precision, dimension(nts) :: sin_shift, cos_shift
        double precision :: sin_tau_center, cos_tau_center
        double precision :: ys_cos_shift, ys_sin_shift, ys_one, sin_shift_one, cos_shift_one
        double precision :: A, B
        !---------------------------------------------------------------------------------

        dz = omega * (ts - tau)
        weights = exp(-c*dz**2)

        sum_w = sum(weights)
        Neff_loc = sum_w**2 / sum(weights**2)

        if (Neff_loc <= Neff) then
            a0 = -99999.
            a1 = -99999.
            a2 = -99999.
        else
            sin_basis = sin(omega*ts)
            cos_basis = cos(omega*ts)
            one_v = 1

            sin_one = sum(sin_basis*one_v*weights) / sum_w
            cos_one = sum(cos_basis*one_v*weights) / sum_w
            sin_cos = sum(sin_basis*cos_basis*weights) / sum_w
            sin_sin = sum(sin_basis*sin_basis*weights) / sum_w
            cos_cos = sum(cos_basis*cos_basis*weights) / sum_w

            numerator = 2*(sin_cos - sin_one*cos_one)
            denominator = (cos_cos - cos_one**2) - (sin_sin - sin_one**2)
            time_shift = atan2(numerator, denominator) / (2*omega)

            sin_shift = sin(omega*(ts - time_shift))
            cos_shift = cos(omega*(ts - time_shift))
            sin_tau_center = sin(omega*(time_shift - tau))
            cos_tau_center = cos(omega*(time_shift - tau))

            ys_cos_shift = sum(pd_ys*cos_shift*weights) / sum_w
            ys_sin_shift = sum(pd_ys*sin_shift*weights) / sum_w
            ys_one = sum(pd_ys*one_v*weights) / sum_w
            sin_shift_one = sum(sin_shift*one_v*weights) / sum_w
            cos_shift_one = sum(cos_shift*one_v*weights) / sum_w

            A = 2*(ys_cos_shift - ys_one*cos_shift_one)
            B = 2*(ys_sin_shift - ys_one*sin_shift_one)

            a0 = ys_one
            a1 = cos_tau_center*A - sin_tau_center*B
            a2 = sin_tau_center*A + cos_tau_center*B
        end if

    end subroutine wwa_1g

end module f2py_wwz
