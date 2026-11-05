package com.etablissement.navette

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DirectionsBus
import androidx.compose.material.icons.filled.Group
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.NavType
import androidx.navigation.navArgument
import com.etablissement.navette.ui.NavetteViewModel
import com.etablissement.navette.ui.screens.AddEditTeacherScreen
import com.etablissement.navette.ui.screens.HomeScreen
import com.etablissement.navette.ui.screens.TeachersScreen
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
    const val TEACHERS = "teachers"
    const val ADD_EDIT = "addEdit"
    const val ADD_EDIT_ARG = "addEdit?teacherId={teacherId}"
    fun addEdit(teacherId: String? = null) =
        if (teacherId == null) ADD_EDIT else "$ADD_EDIT?teacherId=$teacherId"
}

@Composable
fun NavetteAppRoot(viewModel: NavetteViewModel = viewModel()) {
    val navController = rememberNavController()

    val bottomItems = listOf(
        Triple(Routes.HOME, "Navette", Icons.Filled.DirectionsBus),
        Triple(Routes.TEACHERS, "Professeurs", Icons.Filled.Group)
    )

    Scaffold(
        bottomBar = {
            val navBackStackEntry by navController.currentBackStackEntryAsState()
            val currentDestination = navBackStackEntry?.destination
            // On masque la barre sur l'écran d'ajout/édition
            val showBar = currentDestination?.route in listOf(Routes.HOME, Routes.TEACHERS)
            if (showBar) {
                NavigationBar {
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
        }
    ) { innerPadding ->
        NavHost(
            navController = navController,
            startDestination = Routes.HOME,
            modifier = Modifier.padding(innerPadding)
        ) {
            composable(Routes.HOME) {
                HomeScreen(viewModel = viewModel)
            }
            composable(Routes.TEACHERS) {
                TeachersScreen(
                    viewModel = viewModel,
                    onAddTeacher = { navController.navigate(Routes.addEdit()) },
                    onEditTeacher = { id -> navController.navigate(Routes.addEdit(id)) }
                )
            }
            composable(
                route = Routes.ADD_EDIT_ARG,
                arguments = listOf(
                    navArgument("teacherId") {
                        type = NavType.StringType
                        nullable = true
                        defaultValue = null
                    }
                )
            ) { backStackEntry ->
                val teacherId = backStackEntry.arguments?.getString("teacherId")
                AddEditTeacherScreen(
                    viewModel = viewModel,
                    teacherId = teacherId,
                    onDone = { navController.popBackStack() }
                )
            }
        }
    }
}
