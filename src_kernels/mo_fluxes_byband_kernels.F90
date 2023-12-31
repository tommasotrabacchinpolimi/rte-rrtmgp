! This code is part of
! RRTM for GCM Applications - Parallel (RRTMGP)
!
! Eli Mlawer and Robert Pincus
! Andre Wehe and Jennifer Delamere
! email:  rrtmgp@aer.com
!
! Copyright 2015,  Atmospheric and Environmental Research and
! Regents of the University of Colorado.  All right reserved.
!
! Use and duplication is permitted under the terms of the
!    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
!
!
! This module is for packaging output quantities from RRTMGP based on spectral flux profiles
!    This implementation provides band-by-band flux profiles
!
module mo_fluxes_byband
  use mo_rte_kind,      only: wp

contains
  subroutine sum_byband(ncol, nlev, ngpt, nbnd, band_lims, spectral_flux, byband_flux) bind (C)
    integer,                               intent(in ) :: ncol, nlev, ngpt, nbnd
    integer,  dimension(2,          nbnd), intent(in ) :: band_lims
    real(wp), dimension(ncol, nlev, ngpt), intent(in ) :: spectral_flux
    real(wp), dimension(ncol, nlev, nbnd), intent(out) :: byband_flux

    integer :: icol, ilev, igpt, ibnd
    !$acc parallel loop collapse(3) copyin(spectral_flux, band_lims) copyout(byband_flux)
    !$omp target teams distribute parallel do collapse(3) map(to:spectral_flux, band_lims) map(from:byband_flux)
    do ibnd = 1, nbnd
      do ilev = 1, nlev
        do icol = 1, ncol
          byband_flux(icol, ilev, ibnd) =  spectral_flux(icol, ilev, band_lims(1, ibnd))
          do igpt = band_lims(1,ibnd)+1, band_lims(2,ibnd)
            byband_flux(icol, ilev, ibnd) = byband_flux(icol, ilev, ibnd) + &
                                            spectral_flux(icol, ilev, igpt)
          end do
        end do
      end do
    enddo
  end subroutine sum_byband
  ! ----------------------------------------------------------------------------
  !
  ! Net flux: Spectral reduction over all points
  !
  subroutine net_byband_full(ncol, nlev, ngpt, nbnd, band_lims, spectral_flux_dn, spectral_flux_up, byband_flux_net) bind (C)
    integer,                               intent(in ) :: ncol, nlev, ngpt, nbnd
    integer,  dimension(2,          nbnd), intent(in ) :: band_lims
    real(wp), dimension(ncol, nlev, ngpt), intent(in ) :: spectral_flux_dn, spectral_flux_up
    real(wp), dimension(ncol, nlev, nbnd), intent(out) :: byband_flux_net

    integer :: icol, ilev, igpt, ibnd

    !$acc parallel loop collapse(3) copyin(spectral_flux_dn, spectral_flux_up, band_lims) copyout(byband_flux_net)
    !$omp target teams distribute parallel do collapse(3) map(to:spectral_flux_dn, spectral_flux_up, band_lims) map(from:byband_flux_net)
    do ibnd = 1, nbnd
      do ilev = 1, nlev
        do icol = 1, ncol
          igpt = band_lims(1,ibnd)
          byband_flux_net(icol, ilev, ibnd) = spectral_flux_dn(icol, ilev, igpt) - &
                                              spectral_flux_up(icol, ilev, igpt)
          do igpt = band_lims(1,ibnd)+1, band_lims(2,ibnd)
            byband_flux_net(icol, ilev, ibnd) = byband_flux_net(icol, ilev, ibnd) + &
                                                spectral_flux_dn(icol, ilev, igpt) - &
                                                spectral_flux_up(icol, ilev, igpt)
          end do
        end do
      end do
    end do
  end subroutine net_byband_full
  ! ----------------------------------------------------------------------------
  subroutine net_byband_precalc(ncol, nlev, nbnd, byband_flux_dn, byband_flux_up, byband_flux_net) bind (C)
    integer,                               intent(in ) :: ncol, nlev, nbnd
    real(wp), dimension(ncol, nlev, nbnd), intent(in ) :: byband_flux_dn, byband_flux_up
    real(wp), dimension(ncol, nlev, nbnd), intent(out) :: byband_flux_net

    byband_flux_net(1:ncol,1:nlev,1:nbnd) = byband_flux_dn(1:ncol,1:nlev,1:nbnd) - byband_flux_up(1:ncol,1:nlev,1:nbnd)
  end subroutine net_byband_precalc
  ! ----------------------------------------------------------------------------

end module mo_fluxes_byband
