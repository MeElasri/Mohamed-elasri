package com.elasri.presence.ui

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.elasri.presence.data.AuthState
import com.elasri.presence.data.LicenseState
import com.elasri.presence.ui.screens.AttendanceScreen
import com.elasri.presence.ui.screens.ClassDetailScreen
import com.elasri.presence.ui.screens.ClassesScreen
import com.elasri.presence.ui.screens.LoginScreen
import com.elasri.presence.ui.screens.NotAuthorizedScreen
import com.elasri.presence.ui.screens.ReportScreen
import com.elasri.presence.ui.screens.SessionsScreen
import com.elasri.presence.ui.screens.StatsScreen
import com.elasri.presence.ui.screens.SubscriptionScreen
import com.elasri.presence.ui.screens.TeachersScreen
import com.elasri.presence.ui.screens.WelcomeScreen

private const val HOME = "home"

@Composable
fun AppNav() {
    val repo = repository()
    val auth by repo.authState.collectAsStateWithLifecycle()
    val license by repo.licenseState.collectAsStateWithLifecycle()
    val nav = rememberNavController()
    val backStack by nav.currentBackStackEntryAsState()
    val classArg = navArgument("classId") { type = NavType.StringType }

    // Signed out (or account disabled) while inside the app → back to the home gate.
    LaunchedEffect(auth) {
        val route = backStack?.destination?.route
        if (auth !is AuthState.Ready && auth !is AuthState.Loading && route != null && route != "welcome" && route != HOME) {
            if (!nav.popBackStack(HOME, inclusive = false)) nav.navigate(HOME)
        }
    }

    // Subscription expired/suspended while inside the app → back to the gate (shows the paywall).
    LaunchedEffect(license) {
        val route = backStack?.destination?.route
        if (license is LicenseState.Blocked && route != null && route != "welcome" && route != HOME) {
            if (!nav.popBackStack(HOME, inclusive = false)) nav.navigate(HOME)
        }
    }

    NavHost(navController = nav, startDestination = "welcome") {

        composable("welcome") {
            WelcomeScreen(onStart = { nav.navigate(HOME) { popUpTo("welcome") { inclusive = true } } })
        }

        // Gate: login, "not authorized", or the class list depending on the account.
        composable(HOME) {
            when (val st = auth) {
                AuthState.Loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator()
                }
                AuthState.SignedOut -> LoginScreen()
                is AuthState.NotAuthorized -> NotAuthorizedScreen(st.email, onSignOut = { repo.signOut() })
                is AuthState.Ready -> when (val lic = license) {
                    LicenseState.Loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator()
                    }
                    is LicenseState.Blocked -> SubscriptionScreen(
                        state = lic,
                        onRetry = { repo.refreshLicense() },
                        onSignOut = { repo.signOut() }
                    )
                    is LicenseState.Valid -> ClassesScreen(
                        profile = st.profile,
                        onOpen = { id -> nav.navigate("class/$id") },
                        onTeachers = { nav.navigate("teachers") },
                        onStats = { nav.navigate("stats") }
                    )
                }
            }
        }

        composable("stats") {
            StatsScreen(onBack = { nav.popBackStack() })
        }

        composable("teachers") {
            TeachersScreen(onBack = { nav.popBackStack() })
        }

        composable("class/{classId}", arguments = listOf(classArg)) { entry ->
            val id = entry.arguments?.getString("classId").orEmpty()
            ClassDetailScreen(
                classId = id,
                onBack = { nav.popBackStack() },
                onTakeAttendance = { nav.navigate("attendance/$id") },
                onHistory = { nav.navigate("sessions/$id") },
                onReports = { nav.navigate("reports/$id") }
            )
        }

        composable(
            "attendance/{classId}?sessionId={sessionId}",
            arguments = listOf(
                classArg,
                navArgument("sessionId") { type = NavType.StringType; nullable = true; defaultValue = null }
            )
        ) { entry ->
            val id = entry.arguments?.getString("classId").orEmpty()
            val sessionId = entry.arguments?.getString("sessionId")?.takeIf { it.isNotBlank() }
            AttendanceScreen(classId = id, sessionId = sessionId, onBack = { nav.popBackStack() })
        }

        composable("sessions/{classId}", arguments = listOf(classArg)) { entry ->
            val id = entry.arguments?.getString("classId").orEmpty()
            SessionsScreen(
                classId = id,
                onBack = { nav.popBackStack() },
                onOpen = { sid -> nav.navigate("attendance/$id?sessionId=$sid") }
            )
        }

        composable("reports/{classId}", arguments = listOf(classArg)) { entry ->
            val id = entry.arguments?.getString("classId").orEmpty()
            ReportScreen(classId = id, onBack = { nav.popBackStack() })
        }
    }
}
