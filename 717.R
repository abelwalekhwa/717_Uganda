#Model Structure EBOLA

library(DiagrammeR)
library(DiagrammeRsvg)
library(rsvg)

diagram_code <- "
digraph Ebola_Model_Structure {

  graph [layout = dot, rankdir = TB, nodesep = 0.5, ranksep = 0.8, fontsize = 30]
  node [shape = box, fontsize = 30]
  edge [penwidth = 2]
  labelloc = t
  fontsize = 30

  subgraph cluster_human {
    labelloc = t
    label = 'Human Population'

    Susceptible [label = 'S']
    Exposed [label = 'E']
    Infectious [label = 'I']
    Recovered [label = 'R']
    Dead [label = 'D (cumulative)']

    Susceptible -> Exposed [label = 'β * I / N', color = red]
    Exposed -> Infectious [label = 'σ']
    Infectious -> Recovered [label = 'γ (1 - CFR)']
    Infectious -> Dead [label = 'γ CFR']
  }

  // Intervention
  Detection [label = 'Detection\\n(detect days)', shape = box]
  Notification [label = 'Notification\\n(notify days)', shape = box]
  Response [label = 'Response\\n(respond days)', shape = box]
  Contact_Tracing [label = 'Contact Tracing\\n(ct_cov coverage)', shape = ellipse, fillcolor = pink, style = filled]

  Detection -> Notification
  Notification -> Response
  Response -> Contact_Tracing

  Contact_Tracing -> Exposed [label = 'ct_cov * remaining E / day', color = blue, style = dashed]
  Contact_Tracing -> Infectious [label = 'ct_cov * remaining I / day', color = blue, style = dashed]

  // Hospitalization (as metric, not compartment)
  Hospitalization [label = 'Hospitalization\\n(hosp_rate * new_inf)', shape = box, fillcolor = orange, style = filled]
  Infectious -> Hospitalization [style = dashed]

  // Positioning
  { rank = source; Detection }
  { rank = same; Detection }
  { rank = sink; Contact_Tracing Hospitalization }
  { rank = same; Contact_Tracing; Hospitalization }
}
"

# Render model diagram
grViz(diagram_code)

# Export SVG
grViz(diagram_code) |>
  export_svg() |>
  charToRaw() |>
  rsvg_svg(file = "Ebola_Model_Structure_Edited.svg")

# Export PNG
rsvg_png(charToRaw(export_svg(grViz(diagram_code))),
         file = "Ebola_Model_Structure_Edited.png",
         width = 3000, height = 4000)

## EBOLA MODEL – REVISED VERSION (28th July 2025)

library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)

### MODEL PARAMETERS ###
N <- 1e6
initial_exp <- 24 # adjusted to fit the total cases
index_case <- 1
days <- 120 # adjusted to cover the outbreak duration
n_sims <- 100
latent_period <- 6 # from paper
infectious_period <- 10 # from paper
R0 <- 1.40 # from paper
hosp_rate <- 0.6 # kept as original, Kabami's paper does not specify
CFR_base <- 0.47 # Kabami et al 2024
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
  CFR_base <= 1)

### SIMULATION FUNCTION ###
simulate_ebola <- function(detect, notify, respond, ct_cov, sim_id, beta = beta_base, CFR = CFR_base) {
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
    lambda <- beta * I[t - 1] / N
    # Transmission
    new_exp <- rbinom(1, S[t - 1], 1 - exp(-lambda))
    new_inf <- rbinom(1, E[t - 1], 1 - exp(-sigma))
    # Recoveries and deaths
    removed <- rbinom(1, I[t - 1], 1 - exp(-gamma))
    new_dea <- rbinom(1, removed, CFR)
    new_rec <- removed - new_dea
    # Contact tracing
    remaining_E <- E[t - 1] + new_exp - new_inf
    remaining_I <- I[t - 1] + new_inf - removed
    traced_E <- if (t >= t_response) rbinom(1, remaining_E, ct_cov) else 0
    traced_I <- if (t >= t_response) rbinom(1, remaining_I, ct_cov) else 0
    traced_total <- traced_total + traced_E + traced_I
    # Compartment updates
    S[t] <- max(0, S[t - 1] - new_exp)
    E[t] <- max(0, remaining_E - traced_E)
    I[t] <- max(0, remaining_I - traced_I)
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
baseline_data <- map_df(1:n_sims, ~ simulate_ebola(43, 1, 7, 0.33, .x))
interv_data <- map_df(1:n_sims, ~ simulate_ebola(7, 1, 7, 0.45, .x))
baseline_data$Scenario <- "Baseline"
interv_data$Scenario <- "7-1-7"
combined <- bind_rows(baseline_data, interv_data)

### VISUALIZATION ###
summary_data <- combined %>%
  pivot_longer(c(I,D), names_to = "Compartment", values_to = "Count") %>%
  group_by(time, Scenario, Compartment) %>%
  dplyr::summarise(
    median = median(Count),
    .groups = "drop")
ggplot(summary_data, aes(time, median, color = Scenario, fill = Scenario)) +
  geom_smooth(se = FALSE, linewidth = 1, method = "loess") +
  facet_wrap(~Compartment, scales = "free_y") +
  labs(title = "Ebola: Number of Deaths and Infected Cases",
       x = "Days", y = "Population") +
  theme_minimal(base_size = 13)

### RESULTS SUMMARY ###
calc_summary <- function(df) {
  sim_summary <- df %>%
    group_by(sim) %>%
    dplyr::summarise(
      cumulative_cases = sum(Incidence),
      cumulative_deaths = sum(Deaths),
      hosp = sum(Hospitalized),
      traced = first(traced),
      .groups = "drop"
    ) %>%
    mutate(
      cost_trace = traced * cost_trace_per_case,
      cost_hosp = hosp * cost_ETU_per_case,
      total_cost = cost_trace + cost_hosp)
  stats <- sim_summary %>%
    select(cumulative_cases, cumulative_deaths, hosp, total_cost) %>%
    pivot_longer(everything(), names_to = "metric", values_to = "value") %>%
    group_by(metric) %>%
    dplyr::summarise(
      mean_val = mean(value),
      median_val = median(value),
      low = quantile(value, 0.025),
      high = quantile(value, 0.975),
      .groups = "drop"
    )
  return(stats)
}

sum_base <- calc_summary(baseline_data)
sum_int <- calc_summary(interv_data)

metrics_order <- c("cumulative_cases", "cumulative_deaths", "hosp", "total_cost")
format_stat <- function(median, low, high, is_cost = FALSE) {
  prefix <- if (is_cost) "$" else ""
  paste0(prefix, format(round(median), big.mark = ","), " (", format(round(low), big.mark = ","), " - ", format(round(high), big.mark = ","), ")")
}

results <- tibble(
  Metric = c("Cases", "Deaths", "Hospitalizations", "Total Cost (USD)"),
  Baseline_Mean = format(round(sum_base$mean_val[match(metrics_order, sum_base$metric)]), big.mark = ","),
  Baseline_Median_UI = mapply(format_stat, sum_base$median_val[match(metrics_order, sum_base$metric)], sum_base$low[match(metrics_order, sum_base$metric)], sum_base$high[match(metrics_order, sum_base$metric)], c(FALSE, FALSE, FALSE, TRUE)),
  `7-1-7_Mean` = format(round(sum_int$mean_val[match(metrics_order, sum_int$metric)]), big.mark = ","),
  `7-1-7_Median_UI` = mapply(format_stat, sum_int$median_val[match(metrics_order, sum_int$metric)], sum_int$low[match(metrics_order, sum_int$metric)], sum_int$high[match(metrics_order, sum_int$metric)], c(FALSE, FALSE, FALSE, TRUE)),
  Averted_Median = format(round(sum_base$median_val[match(metrics_order, sum_base$metric)] - sum_int$median_val[match(metrics_order, sum_int$metric)]), big.mark = ",")
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
    
    ct_test = ct_cov
    
    sim_data <- map_df(1:sims, function(id) {
      
      simulate_ebola(
        
        detect = 10, notify = 1, respond = 1,
        
        ct_cov = ct_test,
        
        sim_id = id,
        
        beta = beta_test,
        
        CFR = CFR_test
        
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
        
        .groups = "drop") %>%
      
      summarise(
        median_cases = median(cases),
        median_deaths = median(deaths),
        total_cost = median(traced * cost_trace_per_case + hosp * cost_ETU_per_case))
    
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
    fill = "R0") +
  
  theme_minimal(base_size = 13)

###################MEASLES###############

library(DiagrammeR)
library(DiagrammeRsvg)
library(rsvg)

diagram_code <- "
digraph Measles_Model_Structure {

  graph [layout = dot, rankdir = TB, nodesep = 0.5, ranksep = 0.8, fontsize = 30]
  node [shape = box, fontsize = 30]
  edge [penwidth = 2]
  labelloc = t
  fontsize = 30

  subgraph cluster_human {
    labelloc = t
    label = 'Human Population'

    Susceptible [label = 'S']
    Exposed [label = 'E']
    Infectious [label = 'I']
    Recovered [label = 'R']
    Dead [label = 'D (cumulative)']

    Susceptible -> Exposed [label = 'β * I / N', color = red]
    Exposed -> Infectious [label = 'σ']
    Infectious -> Recovered [label = 'γ (1 - CFR)']
    Infectious -> Dead [label = 'γ CFR']
  }

  // Interventions
  Detection [label = 'Detection\\n(detect days)', shape = box]
  Notification [label = 'Notification\\n(notify days)', shape = box]
  Response [label = 'Response\\n(respond days)', shape = box]

  Contact_Tracing [label = 'Contact Tracing\\n(ct_cov coverage)', shape = ellipse, fillcolor = pink, style = filled]
  Reactive_Vaccination [label = 'Reactive Vaccination\\n(reactive_cov, reactive_eff)', shape = ellipse, fillcolor = purple, style = filled]
  Education [label = 'Education\\n(edu_red reduction in β)', shape = diamond, fillcolor = yellow, style = filled]
  Nutrition [label = 'Nutrition (Vit A)\\n(nut_red reduction in CFR)', shape = diamond, fillcolor = green, style = filled]

  Detection -> Notification
  Notification -> Response
  Response -> Contact_Tracing
  Response -> Reactive_Vaccination
  Response -> Education
  Response -> Nutrition

  Contact_Tracing -> Exposed [label = 'ct_cov * remaining E / day\\n(to R)', color = blue, style = dashed]
  Contact_Tracing -> Infectious [label = 'ct_cov * remaining I / day\\n(to R)', color = blue, style = dashed]

  Reactive_Vaccination -> Susceptible [label = 'vacc_effect / day\\n(to R)', color = purple, style = dashed, arrowhead = none]

  Education -> Susceptible [style = dashed, label = 'Reduces β', color = yellow, arrowhead = none]

  Nutrition -> Infectious [style = dashed, label = 'Reduces CFR', color = green, arrowhead = none]

  // Hospitalization (as metric, not compartment)
  Hospitalization [label = 'Hospitalization\\n(hosp_rate * new_inf)', shape = box, fillcolor = orange, style = filled]
  Infectious -> Hospitalization [style = dashed]

  // Positioning
  { rank = source; Detection }
  { rank = same; Detection }
  { rank = sink; Contact_Tracing Reactive_Vaccination Education Nutrition Hospitalization }
  { rank = same; Contact_Tracing; Reactive_Vaccination; Education; Nutrition; Hospitalization }
}
"

# Render model diagram
grViz(diagram_code)

# Export SVG
grViz(diagram_code) |>
  export_svg() |>
  charToRaw() |>
  rsvg_svg(file = "Measles_Model_Structure_Edited.svg")

# Export PNG
rsvg_png(charToRaw(export_svg(grViz(diagram_code))),
         file = "Measles_Model_Structure_Edited.png",
         width = 3000, height = 4000)


# MEASLES TRANSMISSION MODEL - NAMISINDWA DISTRICT (28th July 2025)
library(tidyverse)
library(ggplot2)

# Demographic parameters
N <- 44905 # Total population (rounded from 44904.9)
routine_cov <- 0.95 # Adjusted for better fit: higher routine vaccination coverage
routine_eff <- 0.93 # Routine vaccine efficacy

# Initial conditions
initial_inf <- 1 # Initial infectious cases
initial_exp <- 10 # Adjusted for better fit: fewer initial exposed

# Simulation settings
days <- 90 # 3 months
n_sims <- 100 # Number of stochastic simulations

# Disease parameters
latent_period <- 10 # Average latent period (days)
infectious_period <- 7 # Average infectious period (days)
R0 <- 12 # Adjusted for better fit: lower effective R0
hosp_rate <- 0.1 # Adjusted for better fit: lower hospitalization rate
CFR_base <- 0.01 # Adjusted for better fit: lower CFR matching observed data

# Derived parameters
beta_base <- R0 / infectious_period
gamma <- 1 / infectious_period
sigma <- 1 / latent_period

# Economic parameters
cost_vacc_per_person <- 10 # Cost per vaccination (USD)
cost_hosp_per_case <- 90.41 # Cost per hospitalization (USD)
cost_vitA <- 23 # Cost per vitamin A supplementation (USD)
cost_trace_per_case <- 15 # Cost per contact tracing case (USD)

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
  camp_rate <- (N * reactive_cov) / camp_dur # Daily vaccination rate
  
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
    new_inf <- rbinom(1, max(0, E[t-1]), 1 - exp(-sigma))
    
    # Recoveries and deaths
    current_I <- I[t-1] + new_inf
    removed <- rbinom(1, max(0, current_I), 1 - exp(-gamma))
    new_dea <- rbinom(1, removed, cfr_t)
    new_rec <- removed - new_dea
    
    ### CONTACT TRACING ###
    remaining_E <- E[t-1] + new_exp - new_inf
    remaining_I <- current_I - removed
    traced_E <- if (t >= t_response) rbinom(1, max(0, remaining_E), ct_cov) else 0
    traced_I <- if (t >= t_response) rbinom(1, max(0, remaining_I), ct_cov) else 0
    traced_total <- traced_total + traced_E + traced_I
    
    ### COMPARTMENT UPDATES ###
    S[t] <- max(0, S[t-1] - new_exp - vacc_effect)
    E[t] <- max(0, remaining_E - traced_E)
    I[t] <- max(0, remaining_I - traced_I)
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
# Baseline scenario (delayed response, no enhanced interventions)

baseline_data <- map_df(1:n_sims, ~ simulate_measles(
  detect = 4, notify = 0, respond = 16, # Adjusted to match dataset medians
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
  sim_summary <- df %>%
    group_by(sim) %>%
    dplyr::summarise(
      cumulative_cases = sum(Incidence, na.rm = TRUE),
      cumulative_deaths = sum(Deaths, na.rm = TRUE),
      hosp = sum(Hospitalized, na.rm = TRUE),
      traced = max(traced, na.rm = TRUE),
      vaccinated = max(vaccinated, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      cost_trace = traced * cost_trace_per_case,
      cost_vacc = vaccinated * cost_vacc_per_person,
      cost_hosp = hosp * cost_hosp_per_case,
      cost_vitA = cumulative_cases * cost_vitA * 0.5, # 50% coverage
      total_cost = cost_trace + cost_vacc + cost_hosp + cost_vitA
    )
  stats <- sim_summary %>%
    select(cumulative_cases, cumulative_deaths, hosp, total_cost) %>%
    pivot_longer(everything(), names_to = "metric", values_to = "value") %>%
    group_by(metric) %>%
    dplyr::summarise(
      mean_val = mean(value),
      median_val = median(value),
      low = quantile(value, 0.025),
      high = quantile(value, 0.975),
      .groups = "drop"
    )
  return(stats)
}

sum_base <- calc_summary(baseline_data)
sum_int <- calc_summary(interv_data)

metrics_order <- c("cumulative_cases", "cumulative_deaths", "hosp", "total_cost")
format_stat <- function(median, low, high, is_cost = FALSE) {
  prefix <- if (is_cost) "$" else ""
  paste0(prefix, format(round(median), big.mark = ","), " (", format(round(low), big.mark = ","), " - ", format(round(high), big.mark = ","), ")")
}

results <- tibble(
  Metric = c("Cases", "Deaths", "Hospitalizations", "Total Cost (USD)"),
  Baseline_Mean = format(round(sum_base$mean_val[match(metrics_order, sum_base$metric)]), big.mark = ","),
  Baseline_Median_UI = mapply(format_stat, sum_base$median_val[match(metrics_order, sum_base$metric)], sum_base$low[match(metrics_order, sum_base$metric)], sum_base$high[match(metrics_order, sum_base$metric)], c(FALSE, FALSE, FALSE, TRUE)),
  `7-1-7_Mean` = format(round(sum_int$mean_val[match(metrics_order, sum_int$metric)]), big.mark = ","),
  `7-1-7_Median_UI` = mapply(format_stat, sum_int$median_val[match(metrics_order, sum_int$metric)], sum_int$low[match(metrics_order, sum_int$metric)], sum_int$high[match(metrics_order, sum_int$metric)], c(FALSE, FALSE, FALSE, TRUE)),
  Averted_Median = format(round(sum_base$median_val[match(metrics_order, sum_base$metric)] - sum_int$median_val[match(metrics_order, sum_int$metric)]), big.mark = ",")
)
print(results)

# DALY CALCULATION
life_expectancy <- 65  # GBD reference life expectancy at birth (approximate for 2010-2019 studies)
avg_age_death_baseline <- 3.5  # Average age at death for measles (primarily children)
avg_age_death_717 <- 3.5  # Same for intervention
disability_weight <- 0.051  # Disability weight for measles (acute moderate episode, from GBD)
duration_disability <- 14 / 365  # Duration of disability in years

cases_median_base <- sum_base$median_val[sum_base$metric == "cumulative_cases"]
deaths_median_base <- sum_base$median_val[sum_base$metric == "cumulative_deaths"]
cases_median_int <- sum_int$median_val[sum_int$metric == "cumulative_cases"]
deaths_median_int <- sum_int$median_val[sum_int$metric == "cumulative_deaths"]
total_cost_median_base <- sum_base$median_val[sum_base$metric == "total_cost"]
total_cost_median_int <- sum_int$median_val[sum_int$metric == "total_cost"]

# Calculate DALY components
YLL_per_death_baseline <- life_expectancy - avg_age_death_baseline
YLL_per_death_717 <- life_expectancy - avg_age_death_717

YLD_per_case_baseline <- disability_weight * duration_disability
YLD_per_case_717 <- disability_weight * duration_disability

# Calculate total DALYs
DALY_base <- (deaths_median_base * YLL_per_death_baseline) +
  (cases_median_base * YLD_per_case_baseline)

DALY_717 <- (deaths_median_int * YLL_per_death_717) +
  (cases_median_int * YLD_per_case_717)

dalys_averted <- DALY_base - DALY_717

cat("\nDALY Results:\n")
cat("Baseline DALYs:", round(DALY_base, 1), "\n")
cat("Intervention DALYs:", round(DALY_717, 1), "\n")
cat("DALYs averted:", round(dalys_averted, 1), "\n")

cost_diff <- total_cost_median_int - total_cost_median_base
cost_per_daly_averted <- if (dalys_averted > 0) cost_diff / dalys_averted else Inf  # Avoid division by zero

cat("\nCost-effectiveness:\n")
cat("Cost per DALY averted: $", round(cost_per_daly_averted, 2), "\n")

# Cost calculations
baseline_cost_ugx <- 340119010  # DHO's office (GOVT, Implementing partners, Donors)
intervention_cost_ugx <- baseline_cost_ugx + 164638500  # Estimated for 7-1-7 interventions
exchange_rate <- 3650
deaths_averted <- deaths_median_base - deaths_median_int  # Updated to use model values (previously hardcoded 1032)

baseline_cost_usd <- baseline_cost_ugx / exchange_rate
intervention_cost_usd <- intervention_cost_ugx / exchange_rate

cat("\nHardcoded Cost Calculations:\n")
cat("Baseline Cost (USD): $", round(baseline_cost_usd, 2), "\n")
cat("Intervention Cost (USD): $", round(intervention_cost_usd, 2), "\n")
cat("Deaths Averted (from model):", deaths_averted, "\n")



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


#### ANTHRAX MODEL WITH ANIMAL COMPONENT # 28th July 2025


###MODEL STRUCTURE####

library(DiagrammeR)
library(DiagrammeRsvg)
library(rsvg)

diagram_code <- "
digraph Anthrax_Model_Structure {

  graph [layout = dot, rankdir = TB, nodesep = 0.5, ranksep = 0.8, fontsize = 30]
  node [shape = box, fontsize = 30]
  edge [penwidth = 2]
  labelloc = t
  fontsize = 30

  subgraph cluster_human {
    labelloc = t
    label = 'Human Population'

    SusceptibleH [label = 'S_h\\n(high/low risk)']
    ExposedH [label = 'E_h']
    InfectiousH [label = 'I_h']
    RecoveredH [label = 'R_h']
    DeadH [label = 'D_h (cumulative)']

    SusceptibleH -> ExposedH [label = 'base_contact (direct carcass contact)\\n+ env_trans_human * Pathogen', color = red]
    ExposedH -> InfectiousH [label = 'σ_h']
    InfectiousH -> RecoveredH [label = 'γ_h (1 - mortality)']
    InfectiousH -> DeadH [label = 'γ_h * mortality']
  }

  subgraph cluster_animal {
    labelloc = t
    label = 'Animal Population'

    SusceptibleA [label = 'S_a']
    ExposedA [label = 'E_a']
    InfectiousA [label = 'I_a']
    RecoveredA [label = 'R_a']
    DeadA [label = 'Dead (cumulative)']

    SusceptibleA -> ExposedA [label = 'env_trans_animal * Pathogen', color = red]
    ExposedA -> InfectiousA [label = 'σ_a']
    InfectiousA -> RecoveredA [label = 'γ_a (1 - mortality)']
    InfectiousA -> DeadA [label = 'γ_a * mortality']
  }

  subgraph cluster_environment {
    labelloc = t
    label = 'Environment'

    Pathogen [label = 'Pathogen']
  }

  DeadA -> Pathogen [label = 'spore_release', color = green]
  Pathogen -> Pathogen [label = '1 - decay', dir = back, style = dashed]

  // Direct carcass contact to humans
  DeadA -> SusceptibleH [label = 'direct contact\\n(eating/touching carcasses)\\n(high/low risk)', color = red, style = dashed]

  // Interventions
  Detection [label = 'Detection\\n(detect days)', shape = box]
  Notification [label = 'Notification\\n(notify days)', shape = box]
  Response [label = 'Response\\n(respond days)', shape = box]
  Vaccination [label = 'Vaccination\\n(vacc_cov coverage, eff)', shape = ellipse, fillcolor = pink, style = filled]
  Disposal [label = 'Carcass Disposal\\n(disposal_cov coverage)', shape = ellipse, fillcolor = pink, style = filled]

  Detection -> Notification
  Notification -> Response
  Response -> Vaccination [label = 'after vaccine_delay']
  Response -> Disposal

  Vaccination -> SusceptibleA [label = 'camp_rate * eff / camp_dur', color = blue, style = dashed]
  Disposal -> DeadA [label = 'disposal_cov * Dead', color = blue, style = dashed]

  // Positioning
  { rank = source; Detection }
  { rank = same; Detection }
  { rank = sink; Vaccination Disposal }
  { rank = same; Vaccination; Disposal }
}
"

# Render model diagram
grViz(diagram_code)

# Export SVG
grViz(diagram_code) |>
  export_svg() |>
  charToRaw() |>
  rsvg_svg(file = "Anthrax_Model_Structure_Updated.svg")

# Export PNG
rsvg_png(charToRaw(export_svg(grViz(diagram_code))),
         file = "Anthrax_Model_Structure_Updated.png",
         width = 3000, height = 4000)


library(tidyverse)
library(ggplot2)

# MODEL PARAMETERS
N_human <- 20322 # Adjusted to fit affected subcounty population
N_animal <- 68893 # District cattle
prop_high_risk <- 0.3 # Adjusted for handlers
initial_animal_inf <- 1
initial_human_exp <- 0

days <- 720 # Adjusted to cover outbreak duration
n_sims <- 500

# Human disease parameters
human_incubation_mean <- 3.5
human_infectious_mean <- 3.1
human_mortality <- 0.19 # Adjusted to fit CFR

# Animal disease parameters
animal_incubation_mean <- 2.8
animal_infectious_dur <- 4
animal_mortality <- 1.0 # Adjusted, most die in data

# Environmental transmission - Further tuned to prevent explosion
base_contact_high <- 0.00005 # Lowered
base_contact_low <- 0.00001 # Lowered
env_transmission_human <- 1e-9 # Lowered
env_transmission_animal <- 1e-8 # Lowered
pathogen_decay <- 0.05 # Increased
pathogen_capacity <- 1e6 # Lowered
spore_release <- 1e3 # Lowered

# Interventions
vaccine_delay <- 6
vaccine_eff <- 0.50
disposal_cov <- 0.10 # Renamed from eff for clarity
camp_dur <- 10 # New for vaccination spread

# Costs (unchanged)
vaccine_cost <- 5000
treat_cost_per_case <- 275000
disposal_cost <- 50000

# Derived rates
sigma_h <- 1 / human_incubation_mean
gamma_h <- 1 / human_infectious_mean
sigma_a <- 1 / animal_incubation_mean
gamma_a <- 1 / animal_infectious_dur

# SIMULATION FUNCTION #
simulate_anthrax <- function(detect, notify, respond, vacc_cov, disposal_cov, sim_id) {
  t_notify <- detect + notify
  t_response <- t_notify + respond
  
  S_h_high <- S_h_low <- E_h <- I_h <- R_h <- D_h <- numeric(days)
  S_a <- E_a <- I_a <- R_a <- Dead <- numeric(days)
  Pathogen <- numeric(days)
  
  new_human_cases <- human_deaths <- numeric(days)
  new_animal_cases <- animal_deaths <- numeric(days)
  
  vaccinated <- disposed <- 0
  
  S_a[1] <- N_animal - initial_animal_inf
  I_a[1] <- initial_animal_inf
  S_h_high[1] <- round(N_human * prop_high_risk)
  S_h_low[1] <- N_human - S_h_high[1]
  Pathogen[1] <- I_a[1] * spore_release / 10 # Initial scaling
  
  camp_rate <- N_animal * vacc_cov / camp_dur
  
  for (t in 2:days) {
    # Vaccination (spread over camp_dur after delay)
    vacc_today <- 0
    if (t >= t_response + vaccine_delay & t < t_response + vaccine_delay + camp_dur) {
      n_vacc <- floor(min(S_a[t-1], camp_rate))
      vacc_today <- rbinom(1, n_vacc, vaccine_eff)
    }
    vaccinated <- vaccinated + vacc_today
    
    # Disposal (daily on Dead)
    disp <- if (t >= t_response) rbinom(1, Dead[t-1], disposal_cov) else 0
    disposed <- disposed + disp
    
    # Animal dynamics
    lambda_animal <- env_transmission_animal * Pathogen[t-1]
    sus_animals <- max(0, S_a[t-1] - vacc_today)
    prob_inf_a <- 1 - exp(-lambda_animal)
    new_Ea <- rbinom(1, sus_animals, prob_inf_a)
    
    prob_prog_a <- 1 - exp(-sigma_a)
    new_Ia <- rbinom(1, E_a[t-1], prob_prog_a)
    
    prob_remove_a <- 1 - exp(-gamma_a)
    removed_a <- rbinom(1, I_a[t-1], prob_remove_a)
    new_dead <- rbinom(1, removed_a, animal_mortality)
    new_Ra <- removed_a - new_dead
    
    S_a[t] <- max(0, S_a[t-1] - new_Ea - vacc_today)
    E_a[t] <- max(0, E_a[t-1] + new_Ea - new_Ia)
    I_a[t] <- max(0, I_a[t-1] + new_Ia - removed_a)
    Dead[t] <- max(0, Dead[t-1] + new_dead - disp)
    R_a[t] <- R_a[t-1] + new_Ra
    
    new_animal_cases[t] <- new_Ia
    animal_deaths[t] <- new_dead
    
    # Pathogen update
    Pathogen[t] <- (Pathogen[t-1] * (1 - pathogen_decay)) + (new_dead * spore_release)
    Pathogen[t] <- min(Pathogen[t], pathogen_capacity)
    
    # Human dynamics
    env_risk <- env_transmission_human * Pathogen[t-1]
    lambda_high <- base_contact_high + env_risk
    lambda_low <- base_contact_low + env_risk
    
    new_Eh_high <- rbinom(1, S_h_high[t-1], 1 - exp(-lambda_high))
    new_Eh_low <- rbinom(1, S_h_low[t-1], 1 - exp(-lambda_low))
    new_Eh <- new_Eh_high + new_Eh_low
    
    prob_prog_h <- 1 - exp(-sigma_h)
    new_Ih <- rbinom(1, E_h[t-1], prob_prog_h)
    
    prob_remove_h <- 1 - exp(-gamma_h)
    removed_h <- rbinom(1, I_h[t-1], prob_remove_h)
    new_Dh <- rbinom(1, removed_h, human_mortality)
    new_Rh <- removed_h - new_Dh
    
    S_h_high[t] <- max(0, S_h_high[t-1] - new_Eh_high)
    S_h_low[t] <- max(0, S_h_low[t-1] - new_Eh_low)
    E_h[t] <- max(0, E_h[t-1] + new_Eh - new_Ih)
    I_h[t] <- max(0, I_h[t-1] + new_Ih - removed_h)
    D_h[t] <- D_h[t-1] + new_Dh
    R_h[t] <- R_h[t-1] + new_Rh
    
    new_human_cases[t] <- new_Ih
    human_deaths[t] <- new_Dh
  }
  
  tibble(
    time = 1:days,
    Sh = S_h_high + S_h_low, Eh = E_h, Ih = I_h, Rh = R_h, Dh = D_h,
    Sa = S_a, Ea = E_a, Ia = I_a, Ra = R_a, Dead,
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
  detect = 150, notify = 7, respond = 11, # Adjusted delay
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

# PLOTS
# Human
ggplot(summary_data %>% filter(Compartment %in% c("human_cases", "human_deaths")),
       aes(x = time, y = median, color = Scenario)) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~Compartment, scales = "free_y",
             labeller = as_labeller(c(human_cases = "New Human Cases", human_deaths = "Human Deaths"))) +
  labs(title = "Impact on Human Anthrax Outcomes", x = "Day", y = "Count") +
  theme_minimal(base_size = 14) +
  scale_color_brewer(palette = "Set1") +
  theme(legend.position = "bottom")

# Animal
ggplot(summary_data %>% filter(Compartment %in% c("animal_cases", "animal_deaths")),
       aes(x = time, y = median, color = Scenario)) +
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

# DALY CALCULATION #
calculate_dalys <- function(cases, deaths, disability_days = 14) {
  yll <- deaths * (65 - 52) # Years of Life Lost
  yld <- cases * 0.33 * (disability_days / 365) # Years Lived with Disability
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
library(deSolve)  # For ode

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