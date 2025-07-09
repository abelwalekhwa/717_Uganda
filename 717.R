
## EBOLA MODEL – REVISED VERSION (4th July 2025)

library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)

### MODEL PARAMETERS ###
N <- 1e6                    #
initial_exp <- 3
index_case <- 1
days <- 87
n_sims <- 100

latent_period <- 10
infectious_period <- 12
R0 <- 2.1                   
hosp_rate <- 0.6
CFR_base <- 0.4

beta_base <- R0 / infectious_period
gamma <- 1 / infectious_period
sigma <- 1 / latent_period

cost_ETU_per_case <- 284.50
cost_trace_per_case <- 42.15

### PARAMETER CHECK ###
stopifnot(
  latent_period > 0,
  infectious_period > 0,
  R0 > 0,
  hosp_rate <= 1,
  CFR_base <= 1
)

### SIMULATION FUNCTION ###
simulate_ebola <- function(detect, notify, respond, ct_cov, sim_id) {
  S <- E <- I <- R <- D <- numeric(days)
  Inc <- Hosp <- Deaths <- numeric(days)
  traced_total <- 0
  
  # Initial states
  S[1] <- N - initial_exp - index_case
  E[1] <- initial_exp
  I[1] <- index_case
  
  for (t in 2:days) {
    t_notify <- detect + notify
    t_response <- t_notify + respond
    
    lambda <- beta_base * I[t - 1] / N
    
    # Transmission
    new_exp <- rbinom(1, S[t - 1], 1 - exp(-lambda))
    new_inf <- rbinom(1, E[t - 1], 1 - exp(-sigma))
    
    # Recoveries and deaths
    new_dea <- rbinom(1, I[t - 1], CFR_base * gamma)
    new_rec <- rbinom(1, I[t - 1] - new_dea, gamma)
    
    # Contact tracing
    traced_E <- if (t >= t_response) rbinom(1, E[t - 1], ct_cov) else 0
    traced_I <- if (t >= t_response) rbinom(1, I[t - 1], ct_cov) else 0
    traced_total <- traced_total + traced_E + traced_I
    
    # Compartment updates
    S[t] <- max(0, S[t - 1] - new_exp)
    E[t] <- max(0, E[t - 1] + new_exp - new_inf - traced_E)
    I[t] <- max(0, I[t - 1] + new_inf - new_dea - new_rec - traced_I)
    R[t] <- R[t - 1] + new_rec
    D[t] <- D[t - 1] + new_dea
    
    # Output metrics
    Inc[t] <- new_inf
    Hosp[t] <- round(new_inf * hosp_rate)
    Deaths[t] <- new_dea
  }
  
  tibble(
    time = 1:days, S, E, I, R, D,
    Incidence = Inc, Hospitalized = Hosp, Deaths,
    traced = traced_total, sim = sim_id
  )
}

### RUN SCENARIOS ###
baseline_data <- map_df(1:n_sims, ~ simulate_ebola(46, 1, 9, 0.2, .x))
interv_data   <- map_df(1:n_sims, ~ simulate_ebola(7, 1, 7, 0.8, .x))

baseline_data$Scenario <- "Baseline"
interv_data$Scenario <- "7-1-7"
combined <- bind_rows(baseline_data, interv_data)

### VISUALIZATION ###
summary_data <- combined %>%
  pivot_longer(c(I,D), names_to = "Compartment", values_to = "Count") %>%
  group_by(time, Scenario, Compartment) %>%
  dplyr::summarise(
    median = median(Count),
    .groups = "drop"
  )

ggplot(summary_data, aes(time, median, color = Scenario, fill = Scenario)) +
  geom_smooth(se = FALSE, linewidth = 1, method = "loess") +
  facet_wrap(~Compartment, scales = "free_y") +
  labs(
    title = "Ebola: Number of Deaths and Infected Cases",
    x = "Days", y = "Population") +
  theme_minimal(base_size = 13)


### RESULTS SUMMARY ###
calc_summary <- function(df) {
  df %>%
    group_by(sim) %>%
    dplyr::summarise(
      cumulative_cases = sum(Incidence),
      cumulative_deaths = sum(Deaths),
      hosp = sum(Hospitalized),
      traced = first(traced),
      .groups = "drop"
    ) %>%
    dplyr::summarise(
      cases_median = median(cumulative_cases),
      deaths_median = median(cumulative_deaths),
      hosp_median = median(hosp),
      cost_trace = median(traced * cost_trace_per_case),
      cost_hosp = median(hosp * cost_ETU_per_case),
      total_cost = cost_trace + cost_hosp
    )
}

sum_base <- calc_summary(baseline_data)
sum_int <- calc_summary(interv_data)

results <- tibble(
  Metric = c("Cases (median)", "Deaths (median)", "Hospitalizations", "Total Cost (USD)"),
  Baseline = c(
    format(round(sum_base$cases_median), big.mark = ","),
    format(round(sum_base$deaths_median), big.mark = ","),
    format(round(sum_base$hosp_median), big.mark = ","),
    paste0("$", format(round(sum_base$total_cost), big.mark = ","))
  ),
  `7-1-7` = c(
    format(round(sum_int$cases_median), big.mark = ","),
    format(round(sum_int$deaths_median), big.mark = ","),
    format(round(sum_int$hosp_median), big.mark = ","),
    paste0("$", format(round(sum_int$total_cost), big.mark = ","))
  ),
  Averted = c(
    format(round(sum_base$cases_median - sum_int$cases_median), big.mark = ","),
    format(round(sum_base$deaths_median - sum_int$deaths_median), big.mark = ","),
    format(round(sum_base$hosp_median - sum_int$hosp_median), big.mark = ","),
    paste0("$", format(round(sum_base$total_cost - sum_int$total_cost), big.mark = ","))
  )
)

print(results)

### DALYs ###
life_expectancy <- 63
avg_age_death_baseline <- 35
avg_age_death_717 <- 35

disability_weight <- 0.133
duration_disability <- 180 / 365

# Deaths
YLL_per_death_baseline <- (life_expectancy - avg_age_death_baseline)
YLL_per_death_717 <- (life_expectancy - avg_age_death_717)

# YLD
YLD_per_case_baseline <- disability_weight * duration_disability
YLD_per_case_717 <- disability_weight * duration_disability

# DALYs
deaths_baseline <- sum_base$deaths_median
cases_baseline <- sum_base$cases_median

deaths_717 <- sum_int$deaths_median
cases_717 <- sum_int$cases_median

DALY_base <- (deaths_baseline * YLL_per_death_baseline) + (cases_baseline * YLD_per_case_baseline)
DALY_717 <- (deaths_717 * YLL_per_death_717) + (cases_717 * YLD_per_case_717)
dalys_averted <- DALY_base - DALY_717

# ICER
cost_diff <- sum_int$total_cost - sum_base$total_cost
cost_per_death_averted <- cost_diff / (deaths_baseline - deaths_717)
ICER <- cost_diff / dalys_averted

# Output
cat("DALYs averted:", round(dalys_averted, 2), "\n")
cat("Cost per death averted (USD):", round(cost_per_death_averted, 2), "\n")
cat("ICER (USD per DALY averted):", round(ICER, 2), "\n")

##### Sensitivity Analysis

run_sensitivity_analysis <- function(
    R0_vals = c(1.8, 2.1, 2.4),
    CFR_vals = c(0.3, 0.4, 0.5),
    ct_cov_vals = c(0.6, 0.8, 1.0),
    sims = 50) {
  results <- expand.grid(R0 = R0_vals, CFR = CFR_vals, ct_cov = ct_cov_vals)
  
  sensitivity_output <- purrr::pmap_dfr(results, function(R0, CFR, ct_cov) {
    beta_test <- R0 / infectious_period
    CFR_test <- CFR
    ct_test <- ct_cov
    
    sim_data <- map_df(1:sims, function(id) {
      simulate_ebola(
        detect = 10, notify = 1, respond = 1,
        ct_cov = ct_test,
        sim_id = id
      ) %>%
        mutate(R0 = R0, CFR = CFR_test, ct_cov = ct_test)
    })
    
    summary <- sim_data %>%
      group_by(sim) %>%
      summarise(
        cases = sum(Incidence),
        deaths = sum(Deaths),
        hosp = sum(Hospitalized),
        traced = first(traced),
        .groups = "drop"
      ) %>%
      summarise(
        median_cases = median(cases),
        median_deaths = median(deaths),
        total_cost = median(traced * cost_trace_per_case + hosp * cost_ETU_per_case)
      )
    
    # DALY estimation for this run
    YLL <- summary$median_deaths * (life_expectancy - avg_age_death_717)
    YLD <- summary$median_cases * disability_weight * duration_disability
    DALYs <- YLL + YLD
    
    tibble(
      R0 = R0,
      CFR = CFR_test,
      ct_cov = ct_test,
      Cases = round(summary$median_cases),
      Deaths = round(summary$median_deaths),
      Total_Cost_USD = round(summary$total_cost),
      DALYs = round(DALYs),
      Cost_per_DALY = round(summary$total_cost / DALYs, 2)
    )
  })
  
  return(sensitivity_output)
}

# Run the sensitivity analysis
sensitivity_results <- run_sensitivity_analysis()

# Print summary
print(sensitivity_results)

# Plot
ggplot(sensitivity_results, aes(x = factor(ct_cov), y = Deaths, fill = factor(R0))) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~CFR, labeller = label_both) +
  labs(
    title = "Sensitivity Analysis: Deaths under Varying R0, CFR, ct_cov",
    x = "Contact Tracing Coverage", y = "Median Deaths",
    fill = "R0"
  ) +
  theme_minimal(base_size = 13)

# MEASLES TRANSMISSION MODEL - NAMISINDWA DISTRICT (4th July 2025)

# MEASLES TRANSMISSION MODEL - NAMISINDWA DISTRICT (4th July 2025)
library(tidyverse)
library(ggplot2)

### MODEL PARAMETERS ###
# Demographic parameters
N <- 44905  # Total population (rounded from 44904.9)
routine_cov <- 0.90  # Routine vaccination coverage for MR1
routine_eff <- 0.93  # Routine vaccine efficacy

# Initial conditions
initial_inf <- 1     # Initial infectious cases
initial_exp <- 40    # Initial exposed individuals

# Simulation settings
days <- 90           # 3 months
n_sims <- 100        # Number of stochastic simulations

# Disease parameters
latent_period <- 10  # Average latent period (days)
infectious_period <- 7  # Average infectious period (days)
R0 <- 12             # Basic reproduction number
hosp_rate <- 0.222   # Hospitalization rate (22.2%)
CFR_base <- 0.037    # Case fatality rate (3.7%)

# Derived parameters
beta_base <- R0 / infectious_period
gamma <- 1 / infectious_period
sigma <- 1 / latent_period

# Economic parameters
cost_vacc_per_person <- 10     # Cost per vaccination (USD)
cost_hosp_per_case <- 90.41    # Cost per hospitalization (USD)
cost_vitA <- 23                # Cost per vitamin A supplementation (USD)
cost_trace_per_case <- 15      # Cost per contact tracing case (USD)

### SIMULATION FUNCTION ###
simulate_measles <- function(detect, notify, respond,
                             reactive_cov, reactive_eff,
                             ct_cov, edu_red, nut_red,
                             sim_id) {
  
  # Initialize compartments and outputs
  S <- E <- I <- R <- numeric(days)
  Inc <- Hosp <- Deaths <- numeric(days)
  vaccinated_total <- traced_total <- 0
  
  # Initial conditions
  R0_init <- round(N * routine_cov * routine_eff)
  S[1] <- N - R0_init - initial_inf - initial_exp
  E[1] <- initial_exp
  I[1] <- initial_inf
  R[1] <- R0_init
  
  # Campaign parameters
  camp_dur <- 10 # Duration of vaccination campaign (days)
  camp_rate <- (N * reactive_cov) / camp_dur  # Daily vaccination rate
  
  # Main simulation loop
  for (t in 2:days) {
    # Calculate intervention timing
    t_notify <- detect + notify
    t_response <- t_notify + respond
    
    ### REACTIVE VACCINATION ###
    vacc_today <- if (t >= t_response && t < t_response + camp_dur)
      min(S[t-1], camp_rate) else 0
    vacc_effect <- round(vacc_today * reactive_eff)
    vaccinated_total <- vaccinated_total + vacc_effect
    
    ### INTERVENTION EFFECTS ###
    # Reduced transmission from education
    beta_t <- if (t >= t_response) beta_base * (1 - edu_red) else beta_base
    
    # Reduced CFR from nutrition (vitamin A)
    cfr_t <- if (t >= t_response) CFR_base * (1 - nut_red) else CFR_base
    
    ### DISEASE DYNAMICS ###
    # Force of infection
    lambda <- beta_t * I[t-1] / N
    
    # State transitions
    new_exp <- rbinom(1, max(0, S[t-1] - vacc_effect), 1 - exp(-lambda))
    new_inf <- rbinom(1, max(0, E[t-1]), sigma)
    new_rec <- rbinom(1, max(0, I[t-1]), gamma)
    
    ### CONTACT TRACING ###
    traced_E <- if (t >= t_response) rbinom(1, E[t-1], ct_cov) else 0
    traced_I <- if (t >= t_response) rbinom(1, I[t-1], ct_cov) else 0
    traced_total <- traced_total + traced_E + traced_I
    
    ### MORTALITY ###
    new_dea <- rbinom(1, max(0, I[t-1] - traced_I - new_rec), cfr_t)
    
    ### COMPARTMENT UPDATES ###
    S[t] <- max(0, S[t-1] - new_exp - vacc_effect)
    E[t] <- max(0, E[t-1] + new_exp - new_inf - traced_E)
    I[t] <- max(0, I[t-1] + new_inf - new_rec - new_dea - traced_I)
    R[t] <- R[t-1] + new_rec + vacc_effect + traced_E + traced_I
    
    ### OUTPUT METRICS ###
    Inc[t] <- new_inf
    Hosp[t] <- round(new_inf * hosp_rate)
    Deaths[t] <- new_dea
  }
  
  # Return results as tibble
  tibble(time = 1:days, S, E, I, R,
         Incidence = Inc, Hospitalized = Hosp, Deaths,
         traced = traced_total,
         vaccinated = vaccinated_total,
         sim = sim_id)
}

### RUN SCENARIOS ###

# Baseline scenario (no interventions)
baseline_data <- map_df(1:n_sims, ~ simulate_measles(
  detect = 5, notify = 24, respond = 11,
  reactive_cov = 0, reactive_eff = 0,
  ct_cov = 0, edu_red = 0, nut_red = 0,
  sim_id = .x))

# 7-1-7 intervention scenario
interv_data <- map_df(1:n_sims, ~ simulate_measles(
  detect = 7, notify = 1, respond = 7,
  reactive_cov = 0.96, reactive_eff = 0.93,
  ct_cov = 0.8, edu_red = 0.3, nut_red = 0.5,
  sim_id = .x))

# Label scenarios
baseline_data$Scenario <- "Baseline"
interv_data$Scenario <- "7-1-7"
combined <- bind_rows(baseline_data, interv_data)

### VISUALIZATION ###

# Prepare summary data for plotting
summary_data <- combined %>%
  pivot_longer(cols = S:Deaths, names_to = "Compartment", values_to = "Count") %>%
  group_by(time, Scenario, Compartment) %>%
  dplyr::summarise(
    median = median(Count),
    .groups = "drop"
  )

# Create comparison plot
ggplot(summary_data, aes(time, median, color = Scenario, fill = Scenario)) +
  geom_line() +
  facet_wrap(~Compartment, scales = "free_y", ncol = 2) +
  labs(title = "Measles Transmission: Baseline vs 7-1-7 Response",
       x = "Day", y = "Count") +
  theme_minimal() +
  theme(legend.position = "bottom")

### RESULTS SUMMARY ###

calc_summary <- function(df) {
  df %>%
    group_by(sim) %>%
    dplyr::summarise(
      cumulative_cases = sum(Incidence, na.rm = TRUE),
      cumulative_deaths = sum(Deaths, na.rm = TRUE),
      cumulative_hosp = sum(Hospitalized, na.rm = TRUE),
      traced = max(traced, na.rm = TRUE),
      vaccinated = max(vaccinated, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::summarise(
      cases_median = median(cumulative_cases),
      deaths_median = median(cumulative_deaths),
      hosp_median = median(cumulative_hosp),
      cost_trace = median(traced * cost_trace_per_case),
      cost_vacc = median(vaccinated * cost_vacc_per_person),
      cost_hosp = median(cumulative_hosp * cost_hosp_per_case),
      cost_vitA = median(cumulative_cases * cost_vitA * 0.5), # 50% coverage
      .groups = "drop"
    ) %>%
    mutate(total_cost = cost_trace + cost_vacc + cost_hosp + cost_vitA)
}

# Calculate scenario summaries
sum_base <- calc_summary(baseline_data)
sum_int <- calc_summary(interv_data)

# Create results table
results <- tibble(
  Metric = c("Cases (median)", "Deaths (median)", "Hospitalizations", "Total Cost (USD)"),
  Baseline = c(
    format(round(sum_base$cases_median), big.mark = ","),
    format(round(sum_base$deaths_median), big.mark = ","),
    format(round(sum_base$hosp_median), big.mark = ","),
    paste0("$", format(round(sum_base$total_cost), big.mark = ","))
  ),
  `7-1-7` = c(
    format(round(sum_int$cases_median), big.mark = ","),
    format(round(sum_int$deaths_median), big.mark = ","),
    format(round(sum_int$hosp_median), big.mark = ","),
    paste0("$", format(round(sum_int$total_cost), big.mark = ","))
  ),
  Averted = c(
    format(round(sum_base$cases_median - sum_int$cases_median), big.mark = ","),
    format(round(sum_base$deaths_median - sum_int$deaths_median), big.mark = ","),
    format(round(sum_base$hosp_median - sum_int$hosp_median), big.mark = ","),
    paste0("$", format(round(sum_base$total_cost - sum_int$total_cost), big.mark = ","))
  )
)

# Print formatted results
print(results)

# DALY CALCULATION
# Parameters
life_expectancy <- 62 # Life expectancy at birth
avg_age_death_baseline <- 3.5 # Average age at death for baseline
avg_age_death_717 <- 3.5 # Same for intervention
disability_weight <- 0.1 # Disability weight for measles
duration_disability <- 14 / 365 # Duration of disability in years

# Calculate DALY components
YLL_per_death_baseline <- life_expectancy - avg_age_death_baseline
YLL_per_death_717 <- life_expectancy - avg_age_death_717

YLD_per_case_baseline <- disability_weight * duration_disability
YLD_per_case_717 <- disability_weight * duration_disability

# Calculate total DALYs
DALY_base <- (sum_base$deaths_median * YLL_per_death_baseline) + 
  (sum_base$cases_median * YLD_per_case_baseline)

DALY_717 <- (sum_int$deaths_median * YLL_per_death_717) + 
  (sum_int$cases_median * YLD_per_case_717)

dalys_averted <- DALY_base - DALY_717

# Output DALY results
cat("\nDALY Results:\n")
cat("Baseline DALYs:", round(DALY_base, 1), "\n")
cat("Intervention DALYs:", round(DALY_717, 1), "\n")
cat("DALYs averted:", round(dalys_averted, 1), "\n")

# Cost-effectiveness
cost_diff <- sum_int$total_cost - sum_base$total_cost
cost_per_daly_averted <- cost_diff / dalys_averted

cat("\nCost-effectiveness:\n")
cat("Cost per DALY averted: $", round(cost_per_daly_averted, 2), "\n")

# Cost calculations
baseline_cost_ugx <- 340119010 # DHO's office (GOVT, Implementing partners, Donors)
intervention_cost_ugx <- baseline_cost_ugx + 164638500 # Estimated for 7-1-7 interventions
exchange_rate <- 3650
deaths_averted <- 1032  # Computed from the model

baseline_cost_usd <- baseline_cost_ugx / exchange_rate
intervention_cost_usd <- intervention_cost_ugx / exchange_rate


## Sensitivity Analysis of the parameters

# Define the simulation function
simulate_measles <- function(detect = 7, notify = 1, respond = 7,
                             reactive_cov, reactive_eff = 0.6,
                             ct_cov, edu_red = 0.3, nut_red,
                             sim_id, R0_value = 12) {
  latent_period <- 10
  infectious_period <- 7
  population <- 44905
  
  beta_base <- R0_value / infectious_period
  
  # Adjusting R0 based on interventions
  effective_R0 <- R0_value * (1 - reactive_cov * reactive_eff) * (1 - ct_cov) * (1 - edu_red)
  
  # Adjusting case fatality based on nutrition
  case_fatality_rate <- 0.02 * (1 - nut_red)
  
  initial_cases <- 10
  cases <- numeric(100)
  deaths <- numeric(100)
  
  cases[1] <- initial_cases
  deaths[1] <- rbinom(1, cases[1], case_fatality_rate)
  
  for (t in 2:100) {
    new_cases <- rpois(1, lambda = cases[t-1] * effective_R0 / infectious_period)
    cases[t] <- min(new_cases, population - sum(cases))  #
    deaths[t] <- rbinom(1, cases[t], case_fatality_rate)
  }
  
  return(data.frame(
    time = 1:100,
    cases = cases,
    deaths = deaths,
    sim = sim_id,
    R0 = R0_value,
    reactive_cov = reactive_cov,
    nut_red = nut_red,
    ct_cov = ct_cov
  ))
}

# Defining parameter grid
param_grid <- expand.grid(
  R0 = c(8, 12, 16),
  reactive_cov = c(0.3, 0.6, 0.9),
  nut_red = c(0, 0.3, 0.6),
  ct_cov = c(0.3, 0.6, 0.9),
  sim_id = 1:5  # multiple runs per scenario for variability
)

# Running simulations over parameter grid
set.seed(123)
sim_results <- purrr::pmap_dfr(param_grid, function(R0, reactive_cov, nut_red, ct_cov, sim_id) {
  simulate_measles(
    reactive_cov = reactive_cov,
    nut_red = nut_red,
    ct_cov = ct_cov,
    sim_id = sim_id,
    R0_value = R0)
})

# Summarizing results
summary_results <- sim_results %>%
  group_by(R0, reactive_cov, nut_red, ct_cov, sim) %>%
  dplyr::summarise(
    total_cases = sum(cases),
    total_deaths = sum(deaths),
    .groups = "drop"
  )

# Ploting results
ggplot(summary_results, aes(x = factor(R0), y = total_deaths, fill = factor(nut_red))) +
  geom_boxplot() +
  facet_grid(ct_cov ~ reactive_cov, labeller = label_both) +
  labs(
    x = "Basic Reproduction Number (R0)",
    y = "Total Deaths (3 months)",
    fill = "Vitamin A Reduction",
    title = "Measles Sensitivity Analysis"
  ) +
  theme_minimal(base_size = 14)


#### ANTHRAX MODEL WITH ANIMAL COMPONENT # 7th July 2025

library(tidyverse)
library(ggplot2)

# MODEL PARAMETERS
N_human <- 103300
N_animal <- 127157
prop_high_risk <- 0.15

initial_animal_inf <- 1
initial_human_exp <- 0

days <- 90
n_sims <- 500

# Human disease parameters
human_incubation_mean <- 3.5
human_incubation_sd <- 1.2
human_infectious_mean <- 3.1
human_mortality <- 0.01

# Animal disease parameters
animal_incubation_mean <- 2.8
animal_incubation_sd <- 0.8
animal_infectious_dur <- 4
animal_mortality <- 0.18

# Environmental transmission
base_contact_high <- 0.0025
base_contact_low <- 0.001
env_transmission_human <- 1e-8
env_transmission_animal <- 3e-7
pathogen_decay <- 0.02
pathogen_capacity <- 1e6

# Interventions
vaccine_delay <- 6
vaccine_eff <- 0.50
disposal_eff <- 0.10

# Costs
vaccine_cost <- 5000
treat_cost_per_case <- 275000
disposal_cost <- 50000

# SIMULATION FUNCTION #
simulate_anthrax <- function(detect, notify, respond, vacc_cov, disposal_cov, sim_id) {
  t_notify <- detect + notify
  t_response <- t_notify + respond
  
  Sh_high <- Sh_low <- Eh <- Ih <- Rh <- Dh <- numeric(days)
  Sa <- Ea <- Ia <- Ra <- Dead <- numeric(days)
  Pathogen <- numeric(days)
  
  new_human_cases <- human_deaths <- numeric(days)
  new_animal_cases <- animal_deaths <- numeric(days)
  
  vaccinated <- disposed <- 0
  
  Sa[1] <- N_animal - initial_animal_inf
  Ia[1] <- initial_animal_inf
  Sh_high[1] <- round(N_human * prop_high_risk)
  Sh_low[1] <- N_human - Sh_high[1]
  Pathogen[1] <- Ia[1] * 0.5
  
  vacc_effect <- function(t) ifelse(t >= t_response, vacc_cov * vaccine_eff, 0)
  
  for (t in 2:days) {
    vacc <- if (t == t_response) round(Sa[t-1] * vacc_cov * vaccine_eff) else 0
    vaccinated <- vaccinated + vacc
    
    disp <- if (t >= t_response) round(Dead[t-1] * disposal_cov * disposal_eff) else 0
    disposed <- disposed + disp
    
    lambda_animal <- env_transmission_animal * Pathogen[t-1]
    sus_animals <- max(0, Sa[t-1] - vacc)
    effective_infection_prob <- (1 - exp(-lambda_animal)) * (1 - vacc_effect(t))
    effective_infection_prob <- min(max(effective_infection_prob, 0), 1)
    
    new_Ea <- rbinom(1, sus_animals, effective_infection_prob)
    p_incub <- pnorm(1, mean = animal_incubation_mean, sd = animal_incubation_sd)
    new_Ia <- rbinom(1, Ea[t-1], min(max(p_incub, 0), 1))
    
    new_dead <- rbinom(1, Ia[t-1], animal_mortality * (1 - vacc_effect(t)))
    recov_prob <- 1 / animal_infectious_dur
    new_Ra <- rbinom(1, Ia[t-1] - new_dead, recov_prob)
    
    Sa[t] <- max(0, Sa[t-1] - new_Ea - vacc)
    Ea[t] <- max(0, Ea[t-1] + new_Ea - new_Ia)
    Ia[t] <- max(0, Ia[t-1] + new_Ia - new_dead - new_Ra)
    Dead[t] <- max(0, Dead[t-1] + new_dead - disp)
    Ra[t] <- Ra[t-1] + new_Ra
    
    new_animal_cases[t] <- new_Ia
    animal_deaths[t] <- new_dead
    
    Pathogen[t] <- (Pathogen[t-1] + Ia[t-1] * 2 + Dead[t-1] * 2) * (1 - pathogen_decay)
    Pathogen[t] <- Pathogen[t] * pathogen_capacity / (Pathogen[t] + pathogen_capacity)
    
    env_risk <- env_transmission_human * Pathogen[t-1]
    lambda_high <- min(base_contact_high + env_risk, 1)
    lambda_low <- min(base_contact_low + env_risk, 1)
    
    new_Eh_high <- rbinom(1, Sh_high[t-1], 1 - exp(-lambda_high))
    new_Eh_low <- rbinom(1, Sh_low[t-1], 1 - exp(-lambda_low))
    new_Eh <- new_Eh_high + new_Eh_low
    
    p_h_incub <- pnorm(1, mean = human_incubation_mean, sd = human_incubation_sd)
    new_Ih <- rbinom(1, Eh[t-1], min(max(p_h_incub, 0), 1))
    
    new_Dh <- rbinom(1, Ih[t-1], human_mortality)
    new_Rh <- rbinom(1, Ih[t-1] - new_Dh, 1 / human_infectious_mean)
    
    Sh_high[t] <- max(0, Sh_high[t-1] - new_Eh_high)
    Sh_low[t] <- max(0, Sh_low[t-1] - new_Eh_low)
    Eh[t] <- max(0, Eh[t-1] + new_Eh - new_Ih)
    Ih[t] <- max(0, Ih[t-1] + new_Ih - new_Dh - new_Rh)
    Dh[t] <- Dh[t-1] + new_Dh
    Rh[t] <- Rh[t-1] + new_Rh
    
    new_human_cases[t] <- new_Ih
    human_deaths[t] <- new_Dh
  }
  
  tibble(
    time = 1:days,
    Sh = Sh_high + Sh_low, Eh, Ih, Rh, Dh,
    Sa, Ea, Ia, Ra, Dead,
    Pathogen,
    human_cases = new_human_cases,
    human_deaths,
    animal_cases = new_animal_cases,
    animal_deaths = animal_deaths,
    vaccinated = vaccinated,
    disposed = disposed,
    sim = sim_id
  )
}

# RUN SIMULATIONS #
set.seed(123)

baseline_data <- map_df(1:n_sims, ~ simulate_anthrax(
  detect = 9, notify = 7, respond = 11,
  vacc_cov = 0, disposal_cov = 0,
  sim_id = .x
))

interv_data <- map_df(1:n_sims, ~ simulate_anthrax(
  detect = 7, notify = 1, respond = 7,
  vacc_cov = 0.40, disposal_cov = 0.45,
  sim_id = .x
))

baseline_data$Scenario <- "Baseline"
interv_data$Scenario <- "Intervention"
combined <- bind_rows(baseline_data, interv_data)

# SUMMARY DATA FOR PLOTTING #
summary_data <- combined %>%
  pivot_longer(cols = c("human_cases", "human_deaths", "animal_cases", "animal_deaths"),
               names_to = "Compartment", values_to = "Count") %>%
  group_by(time, Scenario, Compartment) %>%
  dplyr::summarise(median = median(Count), .groups = "drop")

# PLOTS #
# Human
ggplot(filter(summary_data, Compartment %in% c("human_cases", "human_deaths")),
       aes(x = time, y = median, color = Scenario, fill = Scenario)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~Compartment, scales = "free_y",
             labeller = as_labeller(c(human_cases = "New Human Cases", human_deaths = "Human Deaths"))) +
  labs(title = "Impact on Human Anthrax Outcomes", x = "Day", y = "Count") +
  theme_minimal(base_size = 14) +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "bottom")

# Animal
ggplot(filter(summary_data, Compartment %in% c("animal_cases", "animal_deaths")),
       aes(x = time, y = median, color = Scenario, fill = Scenario)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~Compartment, scales = "free_y",
             labeller = as_labeller(c(animal_cases = "New Animal Cases", animal_deaths = "Animal Deaths"))) +
  labs(title = "Impact on Animal Anthrax Outcomes", x = "Day", y = "Count") +
  theme_minimal(base_size = 14) +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "bottom")

# ECONOMIC SUMMARY#
calc_summary <- function(df) {
  df %>%
    group_by(sim, Scenario) %>%
    dplyr::summarise(
      human_cases = sum(human_cases),
      human_deaths = sum(human_deaths),
      animal_cases = sum(animal_cases),
      animal_deaths = sum(animal_deaths),
      vaccinated = max(vaccinated),
      disposed = max(disposed),
      .groups = "drop"
    )
}

econ_summary <- bind_rows(calc_summary(baseline_data), calc_summary(interv_data)) %>%
  group_by(Scenario) %>%
  dplyr::summarise(
    mean_human_cases = mean(human_cases),
    mean_human_deaths = mean(human_deaths),
    mean_animal_cases = mean(animal_cases),
    mean_animal_deaths = mean(animal_deaths),
    mean_vaccinated = mean(vaccinated),
    mean_disposed = mean(disposed)
  ) %>%
  mutate(
    vacc_cost = mean_vaccinated * vaccine_cost,
    treat_cost = mean_human_cases * treat_cost_per_case,
    disposal_cost_total = mean_disposed * disposal_cost,
    total_cost = vacc_cost + treat_cost + disposal_cost_total
  )

print(econ_summary)

# DALY CALCULATION  #
calculate_dalys <- function(cases, deaths, disability_days = 14) {
  yll <- deaths * (65 - 52)  # Years of Life Lost
  yld <- cases * 0.33 * (disability_days / 365)  # Years Lived with Disability
  yll + yld
}

# DALYs per simulation
dalys_df <- bind_rows(calc_summary(baseline_data), calc_summary(interv_data)) %>%
  mutate(
    dalys = calculate_dalys(human_cases, human_deaths)) %>%
  group_by(Scenario) %>%
  dplyr::summarise(mean_dalys = mean(dalys))

print(dalys_df)

## Sensitivity Analysis for Anthrax Model

library(reshape2)

# SEIR model with vaccination and carcass disposal
seir_model <- function(time, state, parameters) {
  with(as.list(c(state, parameters)), {
    
    # Adjust beta to reflect vaccination coverage
    effective_beta <- beta * (1 - v_coverage)
    
    dS <- -effective_beta * S * I
    dE <- effective_beta * S * I - sigma * E
    dI <- sigma * E - gamma * I - disposal_rate * I
    dR <- gamma * I
    dD <- disposal_rate * I  # Removed infected carcasses
    
    return(list(c(dS, dE, dI, dR, dD)))
  })
}

# Baseline parameters
parameters <- list(
  beta = 0.5,          # Infection rate (to be derived from R0)
  sigma = 1/5.2,       # Incubation rate (1/latent period)
  gamma = 1/10,        # Recovery rate (1/infectious period)
  v_coverage = 0.5,    # Vaccination coverage (50%)
  disposal_rate = 0.1  # Carcass disposal rate
  )

# Initial state
initial_state <- c(S = 0.99, E = 0.005, I = 0.005, R = 0, D = 0)

# Time points
times <- seq(0, 200, by = 1)

# Sensitivity ranges
r0_values <- seq(1.2, 5, by = 0.4)
vacc_coverage_values <- seq(0, 0.9, by = 0.1)
disposal_values <- seq(0, 0.5, by = 0.05)

# Store results
results <- data.frame()

# Sensitivity loop
for (r0 in r0_values) {
  for (v in vacc_coverage_values) {
    for (d in disposal_values) {
      
      gamma <- parameters$gamma
      beta <- r0 * gamma
      
      params <- c(
        beta = beta,
        sigma = parameters$sigma,
        gamma = gamma,
        v_coverage = v,
        disposal_rate = d)
      
      out <- ode(y = initial_state, times = times, func = seir_model, parms = params)
      out_df <- as.data.frame(out)
      final_infected <- max(out_df$I + out_df$R + out_df$D)
      
      results <- rbind(results, data.frame(
        R0 = r0,
        Vaccination_Coverage = v,
        Disposal_Rate = d,
        Final_Size = final_infected
      ))
    }
  }
}

#Plot 1: Final size vs. R0 for varying vaccination coverage
ggplot(results, aes(x = R0, y = Final_Size, color = factor(Vaccination_Coverage))) +
  geom_line(size = 1) +
  facet_wrap(~ Disposal_Rate, labeller = label_both) +
  labs(title = "Sensitivity to R0, Vaccination and Disposal",
       y = "Final Outbreak Size", color = "Vacc. Coverage") +
  theme_minimal()
