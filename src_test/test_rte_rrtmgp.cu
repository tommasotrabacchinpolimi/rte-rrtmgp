/*
 * This file is a stand-alone executable developed for the
 * testing of the C++ interface to the RTE+RRTMGP radiation code.
 *
 * It is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/algorithm/string.hpp>
#include <chrono>
#include <iomanip>
#include <cuda_profiler_api.h>


#include "Status.h"
#include "Netcdf_interface.h"
#include "Array.h"
#include "Radiation_solver.h"
#include "Gas_concs.h"
#include "Types.h"
#include "Mem_pool_gpu.h"


void read_and_set_vmr(
        const std::string& gas_name, const int n_col_x, const int n_col_y, const int n_lay,
        const Netcdf_handle& input_nc, Gas_concs& gas_concs)
{
    const std::string vmr_gas_name = "vmr_" + gas_name;

    if (input_nc.variable_exists(vmr_gas_name))
    {
        std::map<std::string, int> dims = input_nc.get_variable_dimensions(vmr_gas_name);
        const int n_dims = dims.size();

        if (n_dims == 0)
        {
            gas_concs.set_vmr(gas_name, input_nc.get_variable<Float>(vmr_gas_name));
        }
        else if (n_dims == 1)
        {
            if (dims.at("lay") == n_lay)
                gas_concs.set_vmr(gas_name,
                        Array<Float,1>(input_nc.get_variable<Float>(vmr_gas_name, {n_lay}), {n_lay}));
            else
                throw std::runtime_error("Illegal dimensions of gas \"" + gas_name + "\" in input");
        }
        else if (n_dims == 3)
        {
            if (dims.at("lay") == n_lay && dims.at("y") == n_col_y && dims.at("x") == n_col_x)
                gas_concs.set_vmr(gas_name,
                        Array<Float,2>(input_nc.get_variable<Float>(vmr_gas_name, {n_lay, n_col_y, n_col_x}), {n_col_x * n_col_y, n_lay}));
            else
                throw std::runtime_error("Illegal dimensions of gas \"" + gas_name + "\" in input");
        }
    }
    else
    {
        Status::print_warning("Gas \"" + gas_name + "\" not available in input file.");
    }
}


void configure_memory_pool(int nlays, int ncols, int nchunks, int ngpts, int nbnds)
{
    /* Heuristic way to set up memory pool queues */
    std::map<std::size_t, std::size_t> pool_queues = {
        {64, 20},
        {128, 20},
        {256, 10},
        {512, 10},
        {1024, 5},
        {2048, 5},
        {nchunks * ngpts * sizeof(Float), 16},
        {nchunks * nbnds * sizeof(Float), 16},
        {(nlays + 1) * ncols * sizeof(Float), 14},
        {(nlays + 1) * nchunks * sizeof(Float), 10},
        {(nlays + 1) * nchunks * nbnds * sizeof(Float), 4},
        {(nlays + 1) * nchunks * ngpts * sizeof(int)/2, 6},
        {(nlays + 1) * nchunks * ngpts * sizeof(Float), 18}
    };

    #ifdef GPU_MEM_POOL
    Memory_pool_gpu::init_instance(pool_queues);
    #endif
}

bool parse_command_line_options(
        std::map<std::string, std::pair<bool, std::string>>& command_line_options,
        int argc, char** argv)
{
    for (int i=1; i<argc; ++i)
    {
        std::string argument(argv[i]);
        boost::trim(argument);

        if (argument == "-h" || argument == "--help")
        {
            Status::print_message("Possible usage:");
            for (const auto& clo : command_line_options)
            {
                std::ostringstream ss;
                ss << std::left << std::setw(30) << ("--" + clo.first);
                ss << clo.second.second << std::endl;
                Status::print_message(ss);
            }
            return true;
        }

        // Check if option starts with --
        if (argument[0] != '-' || argument[1] != '-')
        {
            std::string error = argument + " is an illegal command line option.";
            throw std::runtime_error(error);
        }
        else
            argument.erase(0, 2);

        // Check if option has prefix no-
        bool enable = true;
        if (argument[0] == 'n' && argument[1] == 'o' && argument[2] == '-')
        {
            enable = false;
            argument.erase(0, 3);
        }

        if (command_line_options.find(argument) == command_line_options.end())
        {
            std::string error = argument + " is an illegal command line option.";
            throw std::runtime_error(error);
        }
        else
            command_line_options.at(argument).first = enable;
    }

    return false;
}


void print_command_line_options(
        const std::map<std::string, std::pair<bool, std::string>>& command_line_options)
{
    Status::print_message("Solver settings:");
    for (const auto& option : command_line_options)
    {
        std::ostringstream ss;
        ss << std::left << std::setw(20) << (option.first);
        ss << " = " << std::boolalpha << option.second.first << std::endl;
        Status::print_message(ss);
    }
}


void solve_radiation(int argc, char** argv)
{
    Status::print_message("###### Starting RTE+RRTMGP solver ######");

    ////// FLOW CONTROL SWITCHES //////
    // Parse the command line options.
    std::map<std::string, std::pair<bool, std::string>> command_line_options {
        {"shortwave"        , { true,  "Enable computation of shortwave radiation."}},
        {"longwave"         , { true,  "Enable computation of longwave radiation." }},
        {"fluxes"           , { true,  "Enable computation of fluxes."             }},
        {"cloud-optics"     , { false, "Enable cloud optics."                      }},
        {"output-optical"   , { false, "Enable output of optical properties."      }},
        {"output-bnd-fluxes", { false, "Enable output of band fluxes."             }},
        {"timings"          , { false, "Repeat computation 10x for run times."     }} };

    if (parse_command_line_options(command_line_options, argc, argv))
        return;

    const bool switch_shortwave         = command_line_options.at("shortwave"        ).first;
    const bool switch_longwave          = command_line_options.at("longwave"         ).first;
    const bool switch_fluxes            = command_line_options.at("fluxes"           ).first;
    const bool switch_cloud_optics      = command_line_options.at("cloud-optics"     ).first;
    const bool switch_output_optical    = command_line_options.at("output-optical"   ).first;
    const bool switch_output_bnd_fluxes = command_line_options.at("output-bnd-fluxes").first;
    const bool switch_timings           = command_line_options.at("timings"          ).first;

    // Print the options to the screen.
    print_command_line_options(command_line_options);


    ////// READ THE ATMOSPHERIC DATA //////
    Status::print_message("Reading atmospheric input data from NetCDF.");

    Netcdf_file input_nc("rte_rrtmgp_input.nc", Netcdf_mode::Read);

    const int n_col_x = input_nc.get_dimension_size("x");
    const int n_col_y = input_nc.get_dimension_size("y");
    const int n_lay = input_nc.get_dimension_size("lay");
    const int n_lev = input_nc.get_dimension_size("lev");
    const int n_col = n_col_x * n_col_y;
    // Read the atmospheric fields.
    Array<Float,2> p_lay(input_nc.get_variable<Float>("p_lay", {n_lay, n_col_y, n_col_x}), {n_col, n_lay});
    Array<Float,2> t_lay(input_nc.get_variable<Float>("t_lay", {n_lay, n_col_y, n_col_x}), {n_col, n_lay});
    Array<Float,2> p_lev(input_nc.get_variable<Float>("p_lev", {n_lev, n_col_y, n_col_x}), {n_col, n_lev});
    Array<Float,2> t_lev(input_nc.get_variable<Float>("t_lev", {n_lev, n_col_y, n_col_x}), {n_col, n_lev});

    // Fetch the col_dry in case present.
    Array<Float,2> col_dry;
    if (input_nc.variable_exists("col_dry"))
    {
        col_dry.set_dims({n_col, n_lay});
        col_dry = std::move(input_nc.get_variable<Float>("col_dry", {n_lay, n_col_y, n_col_x}));
    }

    // Create container for the gas concentrations and read gases.
    Gas_concs gas_concs;

    read_and_set_vmr("h2o", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("co2", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("o3" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("n2o", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("co" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("ch4", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("o2" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("n2" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);

    read_and_set_vmr("ccl4"   , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc11"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc12"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cfc22"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc143a", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc125" , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc23"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc32"  , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("hfc134a", n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("cf4"    , n_col_x, n_col_y, n_lay, input_nc, gas_concs);
    read_and_set_vmr("no2"    , n_col_x, n_col_y, n_lay, input_nc, gas_concs);

    Array<Float,2> lwp;
    Array<Float,2> iwp;
    Array<Float,2> rel;
    Array<Float,2> rei;

    if (switch_cloud_optics)
    {
        lwp.set_dims({n_col, n_lay});
        lwp = std::move(input_nc.get_variable<Float>("lwp", {n_lay, n_col_y, n_col_x}));

        iwp.set_dims({n_col, n_lay});
        iwp = std::move(input_nc.get_variable<Float>("iwp", {n_lay, n_col_y, n_col_x}));

        rel.set_dims({n_col, n_lay});
        rel = std::move(input_nc.get_variable<Float>("rel", {n_lay, n_col_y, n_col_x}));

        rei.set_dims({n_col, n_lay});
        rei = std::move(input_nc.get_variable<Float>("rei", {n_lay, n_col_y, n_col_x}));
    }


    ////// CREATE THE OUTPUT FILE //////
    // Create the general dimensions and arrays.
    Status::print_message("Preparing NetCDF output file.");

    Netcdf_file output_nc("rte_rrtmgp_output.nc", Netcdf_mode::Create);
    output_nc.add_dimension("x", n_col_x);
    output_nc.add_dimension("y", n_col_y);
    output_nc.add_dimension("lay", n_lay);
    output_nc.add_dimension("lev", n_lev);
    output_nc.add_dimension("pair", 2);

    auto nc_lay = output_nc.add_variable<Float>("p_lay", {"lay", "y", "x"});
    auto nc_lev = output_nc.add_variable<Float>("p_lev", {"lev", "y", "x"});

    nc_lay.insert(p_lay.v(), {0, 0, 0});
    nc_lev.insert(p_lev.v(), {0, 0, 0});

    int ngpts = 0;
    int nbnds = 0;
    if (switch_longwave)
    {
        Netcdf_file coef_nc_lw("coefficients_lw.nc", Netcdf_mode::Read);
        nbnds = std::max(coef_nc_lw.get_dimension_size("bnd"), nbnds);
        ngpts = std::max(coef_nc_lw.get_dimension_size("gpt"), ngpts);
    }
    if (switch_shortwave)
    {
        Netcdf_file coef_nc_sw("coefficients_sw.nc", Netcdf_mode::Read);
        nbnds = std::max(coef_nc_sw.get_dimension_size("bnd"), nbnds);
        ngpts = std::max(coef_nc_sw.get_dimension_size("gpt"), ngpts);
    }
    configure_memory_pool(n_lay, n_col, 512, ngpts, nbnds);


    ////// RUN THE LONGWAVE SOLVER //////
    if (switch_longwave)
    {
        // Initialize the solver.
        Status::print_message("Initializing the longwave solver.");

        Gas_concs_gpu gas_concs_gpu(gas_concs);
        
        Radiation_solver_longwave rad_lw(gas_concs_gpu, "coefficients_lw.nc", "cloud_coefficients_lw.nc");

        // Read the boundary conditions.
        const int n_bnd_lw = rad_lw.get_n_bnd_gpu();
        const int n_gpt_lw = rad_lw.get_n_gpt_gpu();

        Array<Float,2> emis_sfc(input_nc.get_variable<Float>("emis_sfc", {n_col_y, n_col_x, n_bnd_lw}), {n_bnd_lw, n_col});
        Array<Float,1> t_sfc(input_nc.get_variable<Float>("t_sfc", {n_col_y, n_col_x}), {n_col});

        // Create output arrays.
        Array_gpu<Float,3> lw_tau;
        Array_gpu<Float,3> lay_source;
        Array_gpu<Float,3> lev_source_inc;
        Array_gpu<Float,3> lev_source_dec;
        Array_gpu<Float,2> sfc_source;

        if (switch_output_optical)
        {
            lw_tau        .set_dims({n_col, n_lay, n_gpt_lw});
            lay_source    .set_dims({n_col, n_lay, n_gpt_lw});
            lev_source_inc.set_dims({n_col, n_lay, n_gpt_lw});
            lev_source_dec.set_dims({n_col, n_lay, n_gpt_lw});
            sfc_source    .set_dims({n_col, n_gpt_lw});
        }

        Array_gpu<Float,2> lw_flux_up;
        Array_gpu<Float,2> lw_flux_dn;
        Array_gpu<Float,2> lw_flux_net;

        if (switch_fluxes)
        {
            lw_flux_up .set_dims({n_col, n_lev});
            lw_flux_dn .set_dims({n_col, n_lev});
            lw_flux_net.set_dims({n_col, n_lev});
        }

        Array_gpu<Float,3> lw_bnd_flux_up;
        Array_gpu<Float,3> lw_bnd_flux_dn;
        Array_gpu<Float,3> lw_bnd_flux_net;

        if (switch_output_bnd_fluxes)
        {
            lw_bnd_flux_up .set_dims({n_col, n_lev, n_bnd_lw});
            lw_bnd_flux_dn .set_dims({n_col, n_lev, n_bnd_lw});
            lw_bnd_flux_net.set_dims({n_col, n_lev, n_bnd_lw});
        }


        // Solve the radiation.

        Status::print_message("Solving the longwave radiation.");

        auto run_solver = [&]()
        {
            Array_gpu<Float,2> p_lay_gpu(p_lay);
            Array_gpu<Float,2> p_lev_gpu(p_lev);
            Array_gpu<Float,2> t_lay_gpu(t_lay);
            Array_gpu<Float,2> t_lev_gpu(t_lev);
            Array_gpu<Float,2> col_dry_gpu(col_dry);
            Array_gpu<Float,1> t_sfc_gpu(t_sfc);
            Array_gpu<Float,2> emis_sfc_gpu(emis_sfc);
            Array_gpu<Float,2> lwp_gpu(lwp);
            Array_gpu<Float,2> iwp_gpu(iwp);
            Array_gpu<Float,2> rel_gpu(rel);
            Array_gpu<Float,2> rei_gpu(rei);

            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);

            rad_lw.solve_gpu(
                    switch_fluxes,
                    switch_cloud_optics,
                    switch_output_optical,
                    switch_output_bnd_fluxes,
                    gas_concs_gpu,
                    p_lay_gpu, p_lev_gpu,
                    t_lay_gpu, t_lev_gpu,
                    col_dry_gpu,
                    t_sfc_gpu, emis_sfc_gpu,
                    lwp_gpu, iwp_gpu,
                    rel_gpu, rei_gpu,
                    lw_tau, lay_source, lev_source_inc, lev_source_dec, sfc_source,
                    lw_flux_up, lw_flux_dn, lw_flux_net,
                    lw_bnd_flux_up, lw_bnd_flux_dn, lw_bnd_flux_net);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            Status::print_message("Duration longwave solver: " + std::to_string(duration) + " (ms)");
        };

        // Tuning step;
        run_solver();

        // Profiling step;
        cudaProfilerStart();
        run_solver();
        cudaProfilerStop();

        if (switch_timings)
        {
            constexpr int n_measures=10;
            for (int n=0; n<n_measures; ++n)
                run_solver();
        }

        //// Store the output.
        Status::print_message("Storing the longwave output.");
        Array<Float,3> lw_tau_cpu(lw_tau);
        Array<Float,3> lay_source_cpu(lay_source);
        Array<Float,2> sfc_source_cpu(sfc_source);
        Array<Float,3> lev_source_inc_cpu(lev_source_inc);
        Array<Float,3> lev_source_dec_cpu(lev_source_dec);
        Array<Float,2> lw_flux_up_cpu(lw_flux_up);
        Array<Float,2> lw_flux_dn_cpu(lw_flux_dn);
        Array<Float,2> lw_flux_net_cpu(lw_flux_net);
        Array<Float,3> lw_bnd_flux_up_cpu(lw_bnd_flux_up);
        Array<Float,3> lw_bnd_flux_dn_cpu(lw_bnd_flux_dn);
        Array<Float,3> lw_bnd_flux_net_cpu(lw_bnd_flux_net);

        output_nc.add_dimension("gpt_lw", n_gpt_lw);
        output_nc.add_dimension("band_lw", n_bnd_lw);

        auto nc_lw_band_lims_wvn = output_nc.add_variable<Float>("lw_band_lims_wvn", {"band_lw", "pair"});
        nc_lw_band_lims_wvn.insert(rad_lw.get_band_lims_wavenumber_gpu().v(), {0, 0});

        if (switch_output_optical)
        {
            auto nc_lw_band_lims_gpt = output_nc.add_variable<int>("lw_band_lims_gpt", {"band_lw", "pair"});
            nc_lw_band_lims_gpt.insert(rad_lw.get_band_lims_gpoint_gpu().v(), {0, 0});

            auto nc_lw_tau = output_nc.add_variable<Float>("lw_tau", {"gpt_lw", "lay", "y", "x"});
            nc_lw_tau.insert(lw_tau_cpu.v(), {0, 0, 0, 0});

            auto nc_lay_source     = output_nc.add_variable<Float>("lay_source"    , {"gpt_lw", "lay", "y", "x"});
            auto nc_lev_source_inc = output_nc.add_variable<Float>("lev_source_inc", {"gpt_lw", "lay", "y", "x"});
            auto nc_lev_source_dec = output_nc.add_variable<Float>("lev_source_dec", {"gpt_lw", "lay", "y", "x"});

            auto nc_sfc_source = output_nc.add_variable<Float>("sfc_source", {"gpt_lw", "y", "x"});

            nc_lay_source.insert    (lay_source_cpu.v()    , {0, 0, 0, 0});
            nc_lev_source_inc.insert(lev_source_inc_cpu.v(), {0, 0, 0, 0});
            nc_lev_source_dec.insert(lev_source_dec_cpu.v(), {0, 0, 0, 0});

            nc_sfc_source.insert(sfc_source_cpu.v(), {0, 0, 0});
        }

        if (switch_fluxes)
        {
            auto nc_lw_flux_up  = output_nc.add_variable<Float>("lw_flux_up" , {"lev", "y", "x"});
            auto nc_lw_flux_dn  = output_nc.add_variable<Float>("lw_flux_dn" , {"lev", "y", "x"});
            auto nc_lw_flux_net = output_nc.add_variable<Float>("lw_flux_net", {"lev", "y", "x"});

            nc_lw_flux_up .insert(lw_flux_up_cpu .v(), {0, 0, 0});
            nc_lw_flux_dn .insert(lw_flux_dn_cpu .v(), {0, 0, 0});
            nc_lw_flux_net.insert(lw_flux_net_cpu.v(), {0, 0, 0});

            if (switch_output_bnd_fluxes)
            {
                auto nc_lw_bnd_flux_up  = output_nc.add_variable<Float>("lw_bnd_flux_up" , {"band_lw", "lev", "y", "x"});
                auto nc_lw_bnd_flux_dn  = output_nc.add_variable<Float>("lw_bnd_flux_dn" , {"band_lw", "lev", "y", "x"});
                auto nc_lw_bnd_flux_net = output_nc.add_variable<Float>("lw_bnd_flux_net", {"band_lw", "lev", "y", "x"});

                nc_lw_bnd_flux_up .insert(lw_bnd_flux_up_cpu.v(), {0, 0, 0, 0});
                nc_lw_bnd_flux_dn .insert(lw_bnd_flux_dn_cpu.v(), {0, 0, 0, 0});
                nc_lw_bnd_flux_net.insert(lw_bnd_flux_net_cpu.v(), {0, 0, 0, 0});
            }
        }
    }


    ////// RUN THE SHORTWAVE SOLVER //////
    if (switch_shortwave)
    {
        // Initialize the solver.
        Status::print_message("Initializing the shortwave solver.");


        Gas_concs_gpu gas_concs_gpu(gas_concs);
        Radiation_solver_shortwave rad_sw(gas_concs_gpu, "coefficients_sw.nc", "cloud_coefficients_sw.nc");

        // Read the boundary conditions.
        const int n_bnd_sw = rad_sw.get_n_bnd_gpu();
        const int n_gpt_sw = rad_sw.get_n_gpt_gpu();

        Array<Float,1> mu0(input_nc.get_variable<Float>("mu0", {n_col_y, n_col_x}), {n_col});
        Array<Float,2> sfc_alb_dir(input_nc.get_variable<Float>("sfc_alb_dir", {n_col_y, n_col_x, n_bnd_sw}), {n_bnd_sw, n_col});
        Array<Float,2> sfc_alb_dif(input_nc.get_variable<Float>("sfc_alb_dif", {n_col_y, n_col_x, n_bnd_sw}), {n_bnd_sw, n_col});

        Array<Float,1> tsi_scaling({n_col});
        if (input_nc.variable_exists("tsi"))
        {
            Array<Float,1> tsi(input_nc.get_variable<Float>("tsi", {n_col_y, n_col_x}), {n_col});
            const Float tsi_ref = rad_sw.get_tsi_gpu();
            for (int icol=1; icol<=n_col; ++icol)
                tsi_scaling({icol}) = tsi({icol}) / tsi_ref;
        }
        else
        {
            for (int icol=1; icol<=n_col; ++icol)
                tsi_scaling({icol}) = Float(1.);
        }

        // Create output arrays.
        Array_gpu<Float,3> sw_tau;
        Array_gpu<Float,3> ssa;
        Array_gpu<Float,3> g;
        Array_gpu<Float,2> toa_source;

        if (switch_output_optical)
        {
            sw_tau    .set_dims({n_col, n_lay, n_gpt_sw});
            ssa       .set_dims({n_col, n_lay, n_gpt_sw});
            g         .set_dims({n_col, n_lay, n_gpt_sw});
            toa_source.set_dims({n_col, n_gpt_sw});
        }

        Array_gpu<Float,2> sw_flux_up;
        Array_gpu<Float,2> sw_flux_dn;
        Array_gpu<Float,2> sw_flux_dn_dir;
        Array_gpu<Float,2> sw_flux_net;

        if (switch_fluxes)
        {
            sw_flux_up    .set_dims({n_col, n_lev});
            sw_flux_dn    .set_dims({n_col, n_lev});
            sw_flux_dn_dir.set_dims({n_col, n_lev});
            sw_flux_net   .set_dims({n_col, n_lev});
        }

        Array_gpu<Float,3> sw_bnd_flux_up;
        Array_gpu<Float,3> sw_bnd_flux_dn;
        Array_gpu<Float,3> sw_bnd_flux_dn_dir;
        Array_gpu<Float,3> sw_bnd_flux_net;

        if (switch_output_bnd_fluxes)
        {
            sw_bnd_flux_up    .set_dims({n_col, n_lev, n_bnd_sw});
            sw_bnd_flux_dn    .set_dims({n_col, n_lev, n_bnd_sw});
            sw_bnd_flux_dn_dir.set_dims({n_col, n_lev, n_bnd_sw});
            sw_bnd_flux_net   .set_dims({n_col, n_lev, n_bnd_sw});
        }


        // Solve the radiation.
        Status::print_message("Solving the shortwave radiation.");

        auto run_solver = [&]()
        {
            Array_gpu<Float,2> p_lay_gpu(p_lay);
            Array_gpu<Float,2> p_lev_gpu(p_lev);
            Array_gpu<Float,2> t_lay_gpu(t_lay);
            Array_gpu<Float,2> t_lev_gpu(t_lev);
            Array_gpu<Float,2> col_dry_gpu(col_dry);
            Array_gpu<Float,2> sfc_alb_dir_gpu(sfc_alb_dir);
            Array_gpu<Float,2> sfc_alb_dif_gpu(sfc_alb_dif);
            Array_gpu<Float,1> tsi_scaling_gpu(tsi_scaling);
            Array_gpu<Float,1> mu0_gpu(mu0);
            Array_gpu<Float,2> lwp_gpu(lwp);
            Array_gpu<Float,2> iwp_gpu(iwp);
            Array_gpu<Float,2> rel_gpu(rel);
            Array_gpu<Float,2> rei_gpu(rei);

            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, 0);

            rad_sw.solve_gpu(
                    switch_fluxes,
                    switch_cloud_optics,
                    switch_output_optical,
                    switch_output_bnd_fluxes,
                    gas_concs_gpu,
                    p_lay_gpu, p_lev_gpu,
                    t_lay_gpu, t_lev_gpu,
                    col_dry_gpu,
                    sfc_alb_dir_gpu, sfc_alb_dif_gpu,
                    tsi_scaling_gpu, mu0_gpu,
                    lwp_gpu, iwp_gpu,
                    rel_gpu, rei_gpu,
                    sw_tau, ssa, g,
                    toa_source,
                    sw_flux_up, sw_flux_dn,
                    sw_flux_dn_dir, sw_flux_net,
                    sw_bnd_flux_up, sw_bnd_flux_dn,
                    sw_bnd_flux_dn_dir, sw_bnd_flux_net);

            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float duration = 0.f;
            cudaEventElapsedTime(&duration, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            Status::print_message("Duration shortwave solver: " + std::to_string(duration) + " (ms)");
        };

        // Tuning step;
        run_solver();

        // Profiling step;
        cudaProfilerStart();
        run_solver();
        cudaProfilerStop();

        if (switch_timings)
        {   
            constexpr int n_measures=10;
            for (int n=0; n<n_measures; ++n)
                run_solver();
        }

        // Store the output.
        Status::print_message("Storing the shortwave output.");
        Array<Float,3> sw_tau_cpu(sw_tau);
        Array<Float,3> ssa_cpu(ssa);
        Array<Float,3> g_cpu(g);
        Array<Float,2> toa_source_cpu(toa_source);
        Array<Float,2> sw_flux_up_cpu(sw_flux_up);
        Array<Float,2> sw_flux_dn_cpu(sw_flux_dn);
        Array<Float,2> sw_flux_dn_dir_cpu(sw_flux_dn_dir);
        Array<Float,2> sw_flux_net_cpu(sw_flux_net);
        Array<Float,3> sw_bnd_flux_up_cpu(sw_bnd_flux_up);
        Array<Float,3> sw_bnd_flux_dn_cpu(sw_bnd_flux_dn);
        Array<Float,3> sw_bnd_flux_dn_dir_cpu(sw_bnd_flux_dn_dir);
        Array<Float,3> sw_bnd_flux_net_cpu(sw_bnd_flux_net);

        output_nc.add_dimension("gpt_sw", n_gpt_sw);
        output_nc.add_dimension("band_sw", n_bnd_sw);

        auto nc_sw_band_lims_wvn = output_nc.add_variable<Float>("sw_band_lims_wvn", {"band_sw", "pair"});
        nc_sw_band_lims_wvn.insert(rad_sw.get_band_lims_wavenumber_gpu().v(), {0, 0});

        if (switch_output_optical)
        {
            auto nc_sw_band_lims_gpt = output_nc.add_variable<int>("sw_band_lims_gpt", {"band_sw", "pair"});
            nc_sw_band_lims_gpt.insert(rad_sw.get_band_lims_gpoint_gpu().v(), {0, 0});

            auto nc_sw_tau = output_nc.add_variable<Float>("sw_tau", {"gpt_sw", "lay", "y", "x"});
            auto nc_ssa    = output_nc.add_variable<Float>("ssa"   , {"gpt_sw", "lay", "y", "x"});
            auto nc_g      = output_nc.add_variable<Float>("g"     , {"gpt_sw", "lay", "y", "x"});

            nc_sw_tau.insert(sw_tau_cpu.v(), {0, 0, 0, 0});
            nc_ssa   .insert(ssa_cpu   .v(), {0, 0, 0, 0});
            nc_g     .insert(g_cpu     .v(), {0, 0, 0, 0});

            auto nc_toa_source = output_nc.add_variable<Float>("toa_source", {"gpt_sw", "y", "x"});
            nc_toa_source.insert(toa_source_cpu.v(), {0, 0, 0});
        }

        if (switch_fluxes)
        {
            auto nc_sw_flux_up     = output_nc.add_variable<Float>("sw_flux_up"    , {"lev", "y", "x"});
            auto nc_sw_flux_dn     = output_nc.add_variable<Float>("sw_flux_dn"    , {"lev", "y", "x"});
            auto nc_sw_flux_dn_dir = output_nc.add_variable<Float>("sw_flux_dn_dir", {"lev", "y", "x"});
            auto nc_sw_flux_net    = output_nc.add_variable<Float>("sw_flux_net"   , {"lev", "y", "x"});

            nc_sw_flux_up    .insert(sw_flux_up_cpu    .v(), {0, 0, 0});
            nc_sw_flux_dn    .insert(sw_flux_dn_cpu    .v(), {0, 0, 0});
            nc_sw_flux_dn_dir.insert(sw_flux_dn_dir_cpu.v(), {0, 0, 0});
            nc_sw_flux_net   .insert(sw_flux_net_cpu   .v(), {0, 0, 0});

            if (switch_output_bnd_fluxes)
            {
                auto nc_sw_bnd_flux_up     = output_nc.add_variable<Float>("sw_bnd_flux_up"    , {"band_sw", "lev", "y", "x"});
                auto nc_sw_bnd_flux_dn     = output_nc.add_variable<Float>("sw_bnd_flux_dn"    , {"band_sw", "lev", "y", "x"});
                auto nc_sw_bnd_flux_dn_dir = output_nc.add_variable<Float>("sw_bnd_flux_dn_dir", {"band_sw", "lev", "y", "x"});
                auto nc_sw_bnd_flux_net    = output_nc.add_variable<Float>("sw_bnd_flux_net"   , {"band_sw", "lev", "y", "x"});

                nc_sw_bnd_flux_up    .insert(sw_bnd_flux_up_cpu    .v(), {0, 0, 0, 0});
                nc_sw_bnd_flux_dn    .insert(sw_bnd_flux_dn_cpu    .v(), {0, 0, 0, 0});
                nc_sw_bnd_flux_dn_dir.insert(sw_bnd_flux_dn_dir_cpu.v(), {0, 0, 0, 0});
                nc_sw_bnd_flux_net   .insert(sw_bnd_flux_net_cpu   .v(), {0, 0, 0, 0});
            }
        }
    }

    Status::print_message("###### Finished RTE+RRTMGP cuda solver ######");
}


int main(int argc, char** argv)
{
    try
    {
        solve_radiation(argc, argv);
    }

    // Catch any exceptions and return 1.
    catch (const std::exception& e)
    {
        std::string error = "EXCEPTION: " + std::string(e.what());
        Status::print_message(error);
        return 1;
    }
    catch (...)
    {
        Status::print_message("UNHANDLED EXCEPTION!");
        return 1;
    }

    // Return 0 in case of normal exit.
    return 0;
}
