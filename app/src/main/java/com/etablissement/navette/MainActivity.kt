package com.etablissement.navette

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DirectionsBus
import androidx.compose.material.icons.filled.Groups
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.NavType
import androidx.navigation.navArgument
import com.etablissement.navette.ui.AuthState
import com.etablissement.navette.ui.NavetteViewModel
import com.etablissement.navette.ui.screens.CreateEditGroupScreen
import com.etablissement.navette.ui.screens.GroupsScreen
import com.etablissement.navette.ui.screens.HomeScreen
import com.etablissement.navette.ui.screens.LoginScreen
import com.etablissement.navette.ui.screens.ProfileScreen
import com.etablissement.navette.ui.theme.NavetteProfsTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            NavetteProfsTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    NavetteAppRoot()
                }
            }
        }
    }
}

private object Routes {
    const val HOME = "home"
    const val PROFILE = "profile"
    const val GROUPS = "groups"
    const val GROUP_EDIT_ARG = "groupEdit?groupId={groupId}"
    fun groupEdit(groupId: String? = null) =
        if (groupId == null) "groupEdit" else "groupEdit?groupId=$groupId"

    val bottomRoutes = listOf(HOME, PROFILE, GROUPS)
}

@Composable
fun NavetteAppRoot(viewModel: NavetteViewModel = viewModel()) {
    val authState by viewModel.authState.collectAsStateWithLifecycle()
    val errorMessage by viewModel.errorMessage.collectAsStateWithLifecycle()
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(errorMessage) {
        errorMessage?.let {
            snackbarHostState.showSnackbar(it)
            viewModel.clearError()
        }
    }

    Scaffold(
        snackbarHost = { SnackbarHost(snackbarHostState) }
    ) { outerPadding ->
        Box(modifier = Modifier.padding(outerPadding)) {
            when (authState) {
                is AuthState.Loading -> LoadingScreen()
                is AuthState.SignedOut -> LoginScreen(viewModel)
                is AuthState.SignedIn -> MainApp(viewModel)
            }
        }
    }
}

@Composable
private fun LoadingScreen() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        CircularProgressIndicator()
    }
}

@Composable
private fun MainApp(viewModel: NavetteViewModel) {
    val navController = rememberNavController()

    val bottomItems = listOf(
        Triple(Routes.HOME, "Navette", Icons.Filled.DirectionsBus),
        Triple(Routes.GROUPS, "Groupes", Icons.Filled.Groups),
        Triple(Routes.PROFILE, "Mon profil", Icons.Filled.Person)
    )

    Scaffold(
        bottomBar = {
            val navBackStackEntry by navController.currentBackStackEntryAsState()
            val currentDestination = navBackStackEntry?.destination
            // Masque la barre d'onglets sur l'écran de création/édition de groupe.
            val showBar = currentDestination?.route?.let { it in Routes.bottomRoutes } ?: false
            if (showBar) NavigationBar {
                bottomItems.forEach { (route, label, icon) ->
                    val selected = currentDestination?.hierarchy?.any { it.route == route } == true
                    NavigationBarItem(
                        selected = selected,
                        onClick = {
                            navController.navigate(route) {
                                popUpTo(navController.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = { Icon(icon, contentDescription = label) },
                        label = { Text(label) }
                    )
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Routes.HOME,
            modifier = Modifier.padding(innerPadding),
            // Transitions légères (fondu rapide) : navigation plus fluide entre les onglets.
            enterTransition = { fadeIn(tween(150)) },
            exitTransition = { fadeOut(tween(150)) },
            popEnterTransition = { fadeIn(tween(150)) },
            popExitTransition = { fadeOut(tween(150)) }
        ) {
            composable(Routes.HOME) { HomeScreen(viewModel = viewModel) }
            composable(Routes.GROUPS) {
                GroupsScreen(
                    viewModel = viewModel,
                    onCreateGroup = { navController.navigate(Routes.groupEdit()) },
                    onEditGroup = { id -> navController.navigate(Routes.groupEdit(id)) }
                )
            }
            composable(Routes.PROFILE) { ProfileScreen(viewModel = viewModel) }
            composable(
                route = Routes.GROUP_EDIT_ARG,
                arguments = listOf(
                    navArgument("groupId") {
                        type = NavType.StringType
                        nullable = true
                        defaultValue = null
                    }
                )
            ) { backStackEntry ->
                CreateEditGroupScreen(
                    viewModel = viewModel,
                    groupId = backStackEntry.arguments?.getString("groupId"),
                    onDone = { navController.popBackStack() }
                )
            }
        }
    }
}
