#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de visualisation académique de l'architecture NeuroLite.

Ce script génère une représentation visuelle de haute qualité de l'architecture
universelle d'IA NeuroLite, adaptée aux publications scientifiques et présentations.
Il met en évidence le noyau latent universel qui sert de "lingua franca" entre les
différentes modalités et les interactions entre les composants clés du système.

Caractéristiques:
- Styles visuels multiples (académique, moderne, minimaliste, sombre)
- Support multilingue (français, anglais)
- Fond texturé et composants améliorés avec effets visuels
- Animations et effets visuels avancés (gradient, ombres, etc.)
- Export en haute résolution pour publications
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from pathlib import Path
import os
import platform
import subprocess
from matplotlib.textpath import TextPath
import matplotlib.transforms as transforms
from matplotlib import gridspec

# Configuration pour style académique
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'Palatino'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 11,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11,
    'lines.linewidth': 1.5,
    'text.usetex': False,  # Mettre à True si LaTeX est installé
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    'figure.figsize': (12, 9),  # Figure plus grande pour plus de détails
    'figure.dpi': 150,         # DPI augmenté pour meilleure résolution à l'écran
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5
})

# Fonction pour créer un dégradé de texture de fond
def create_texture_background(ax, style, density=20, alpha=0.05):
    """Crée un fond texturé pour le diagramme selon le style."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    
    if style == "academic":
        # Grille discrète pour style académique
        for x in np.linspace(xmin, xmax, int(density)):
            ax.axvline(x, color='black', alpha=alpha/2, linestyle='-', linewidth=0.5)
        for y in np.linspace(ymin, ymax, int(density/2)):
            ax.axhline(y, color='black', alpha=alpha/2, linestyle='-', linewidth=0.5)
    
    elif style == "modern":
        # Points aléatoires pour style moderne
        n_points = int(density * 5)
        x_points = np.random.uniform(xmin, xmax, n_points)
        y_points = np.random.uniform(ymin, ymax, n_points)
        ax.scatter(x_points, y_points, s=0.5, color='white', alpha=alpha*2)
        
        # Cercles concentriques
        for r in np.linspace(1, max(width, height)/2, 5):
            circle = plt.Circle((xmin + width/2, ymin + height/2), r, 
                              fill=False, alpha=alpha*3, color='white', linestyle='-', linewidth=0.5)
            ax.add_patch(circle)
    
    elif style == "dark":
        # Lignes de flux pour style sombre
        for i in range(int(density/2)):
            x = np.linspace(xmin, xmax, 100)
            y = ymin + height/2 + np.sin(x*np.pi/5 + i) * height/5
            ax.plot(x, y, alpha=alpha*3, color='#4080ff', linewidth=0.7)
            
    elif style == "minimal":
        # Points discrets pour style minimal
        grid_x, grid_y = np.meshgrid(
            np.linspace(xmin, xmax, int(density/2)), 
            np.linspace(ymin, ymax, int(density/2))
        )
        ax.scatter(grid_x.flatten(), grid_y.flatten(), s=0.8, color='#888888', alpha=alpha)

# Fonction pour créer un effet de lumière/brillance
def add_glow_effect(ax, x, y, radius, color, alpha=0.3, n_circles=7):
    """Ajoute un effet de brillance/halo autour d'un point."""
    max_alpha = alpha
    for i in range(n_circles):
        r = radius * (1 - i/n_circles)
        a = max_alpha * (1 - i/n_circles)
        circle = plt.Circle((x, y), r, color=color, alpha=a, fill=True, linewidth=0)
        ax.add_patch(circle)


def create_neurolite_architecture_diagram(save_path="outputs/neurolite_architecture_academic.png", 
                                         style="academic", lang="fr", 
                                         add_texture=True, use_glow_effect=True,
                                         high_quality=True, show_animation=False):
    """
    Crée un diagramme visuel de l'architecture NeuroLite adapté aux publications et présentations.
    
    Args:
        save_path: Chemin où sauvegarder l'image générée
        style: Style visuel ('academic', 'modern', 'minimal', 'dark')
        lang: Langue des annotations ('fr', 'en')
        add_texture: Ajouter une texture de fond selon le style
        use_glow_effect: Ajouter des effets de lueur autour des composants clés
        high_quality: Utiliser des paramètres de haute qualité pour l'export
        show_animation: Animer les connexions entre composants (mode présentation)
        
    Returns:
        Chemin vers l'image générée
    """
    # Créer le répertoire de sortie si nécessaire
    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Définir le style avec options avancées
    if style == "academic":
        plt.style.use('seaborn-v0_8-whitegrid')
        bg_color = "#ffffff"
        text_color = "#000000"
        accent_color = "#3380aa"
        box_alpha = 0.9
        edge_alpha = 0.7
        gradient_style = "radial"  # Dégradé radial pour composants
        texture_density = 18
        texture_alpha = 0.03
        glow_alpha = 0.15
        glow_radius = 1.0
        shadow_offset = 0.04
        dpi_export = 450 if high_quality else 300
        line_style = "solid"        
    elif style == "modern":
        plt.style.use('dark_background')
        bg_color = "#0d1117"  # Fond sombre amélioré
        text_color = "#ffffff"
        accent_color = "#4d80e4"
        box_alpha = 0.85
        edge_alpha = 0.7
        gradient_style = "linear"  # Dégradé linéaire pour composants
        texture_density = 25
        texture_alpha = 0.08
        glow_alpha = 0.3
        glow_radius = 1.3
        shadow_offset = 0.05
        dpi_export = 400 if high_quality else 300
        line_style = "solid"
    elif style == "minimal":
        plt.style.use('ggplot')
        bg_color = "#f8f9fa"
        text_color = "#333333"
        accent_color = "#2c3e50"
        box_alpha = 1.0
        edge_alpha = 0.7
        gradient_style = "none"  # Pas de dégradé pour style minimal
        texture_density = 12
        texture_alpha = 0.02
        glow_alpha = 0.1
        glow_radius = 0.7
        shadow_offset = 0.03
        dpi_export = 400 if high_quality else 300
        line_style = "dashed"
    elif style == "dark":
        plt.style.use('dark_background')
        bg_color = "#111111"
        text_color = "#e0e0e0"
        accent_color = "#8a2be2"  # Violet vif
        box_alpha = 0.85
        edge_alpha = 0.8
        gradient_style = "glow"  # Effet lumineux pour style sombre
        texture_density = 30
        texture_alpha = 0.1
        glow_alpha = 0.4
        glow_radius = 1.5
        shadow_offset = 0.06
        dpi_export = 450 if high_quality else 300
        line_style = "solid"
    else:  # Style académique par défaut
        plt.style.use('seaborn-v0_8-whitegrid')
        bg_color = "#ffffff"
        text_color = "#000000"
        accent_color = "#3380aa"
        box_alpha = 0.9
        edge_alpha = 0.7
        gradient_style = "radial"
        texture_density = 18
        texture_alpha = 0.03
        glow_alpha = 0.15
        glow_radius = 1.0
        shadow_offset = 0.04
        dpi_export = 400 if high_quality else 300
        line_style = "solid"
    
    # Créer le dossier architectures si nécessaire
    output_dir = Path(save_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remplacer le dossier outputs par architectures si nécessaire
    if 'outputs' in str(output_dir):
        arch_dir = Path(str(output_dir).replace('outputs', 'architectures'))
        arch_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_path).replace('outputs', 'architectures')
    
    # Traductions selon la langue avec les détails enrichis du README
    translations = {
        'fr': {
            'title': 'Architecture Universelle NeuroLite',
            'subtitle': 'Noyau Latent Universel et Encodeurs Multimodaux Spécialisés',
            
            # Entrées
            'text_input': 'Entrée Texte',
            'image_input': 'Entrée Image',
            'audio_input': 'Entrée Audio',
            'video_input': 'Entrée Vidéo',
            'graph_input': 'Entrée Graphe',
            
            # Tokenizer
            'tokenizer': 'Tokenizer Multimodal',
            'tokenizer_detail': 'Double Codebook & Quantification Vectorielle Résiduelle',
            
            # Encodeurs
            'text_encoder': 'Encodeur\nTexte',
            'text_encoder_detail': 'Attention & Embeddings Avancés',
            'image_encoder': 'Encodeur\nImage',
            'image_encoder_detail': 'Vision Transformer (ViT)',
            'audio_encoder': 'Encodeur\nAudio',
            'audio_encoder_detail': 'Convolutions & Encodage Spectral',
            'video_encoder': 'Encodeur\nVidéo',
            'video_encoder_detail': 'Traitement Spatio-temporel',
            'graph_encoder': 'Encodeur\nGraphe',
            'graph_encoder_detail': 'Attention sur Relations',
            
            # Core components
            'latent_core': 'Noyau Latent\nUniversel',
            'latent_core_detail': 'Backbone All-MLP & HyperMixer',
            'memory': 'Mémoire\nHiérarchique',
            'memory_detail': 'Court terme, Long terme\nPersistante, Associative',
            'routing': 'Routage\nDynamique',
            'routing_detail': 'Mixture-of-Experts\nActivation Conditionnelle',
            'symbolic': 'Module\nNeurosymbolique',
            'symbolic_detail': 'Moteur de Règles & Réseau Bayésien',
            'cross_modal': 'Attention\nCross-Modale',
            'cross_modal_detail': 'Fusion Adaptative\nInteraction Multimodale',
            'continual': 'Apprentissage\nContinu',
            'continual_detail': 'Réduction d\'Oubli\nAdaptation Progressive',
            
            # Sorties
            'text_output': 'Sortie Texte',
            'image_output': 'Sortie Image',
            'audio_output': 'Sortie Audio',
            'video_output': 'Sortie Vidéo',
            'graph_output': 'Sortie Graphe',
            
            # Caractéristiques
            'complexity': 'Complexité: O(n) - Linéaire en longueur de séquence',
            'params': 'Taille: ~335M paramètres (60% dans les encodeurs)',
            'routing_info': 'Routage conditionnel: n\'active que les composants nécessaires',
            'memory_info': 'Mémoire hiérarchique pour une rétention contextuelle efficace',
            'symbolic_info': 'Composants neurosymboliques pour le raisonnement structuré',
            'backbone_info': 'Architecture hybride avec attention linéaire et MLP-Mixer',
            
            # Légende
            'legend_encoders': 'Encodeurs Modulaires',
            'legend_latent': 'Noyau Latent Universel',
            'legend_memory': 'Mémoire Multi-niveaux',
            'legend_routing': 'Routage Adaptatif',
            'legend_symbolic': 'Module Neurosymbolique',
            'legend_cross_modal': 'Attention Cross-Modale',
            'legend_continual': 'Apprentissage Continu',
            'legend_output': 'Projections de Sortie',
            'legend_tokenizer': 'Tokenizer Multimodal'
        },
        'en': {
            'title': 'NeuroLite Universal Architecture',
            'subtitle': 'Universal Latent Core and Specialized Multimodal Encoders',
            
            # Inputs
            'text_input': 'Text Input',
            'image_input': 'Image Input',
            'audio_input': 'Audio Input',
            'video_input': 'Video Input',
            'graph_input': 'Graph Input',
            
            # Tokenizer
            'tokenizer': 'Multimodal Tokenizer',
            'tokenizer_detail': 'Dual Codebook & Residual Vector Quantization',
            
            # Encoders
            'text_encoder': 'Text\nEncoder',
            'text_encoder_detail': 'Attention & Advanced Embeddings',
            'image_encoder': 'Image\nEncoder',
            'image_encoder_detail': 'Vision Transformer (ViT)',
            'audio_encoder': 'Audio\nEncoder',
            'audio_encoder_detail': 'Convolutions & Spectral Encoding',
            'video_encoder': 'Video\nEncoder',
            'video_encoder_detail': 'Spatiotemporal Processing',
            'graph_encoder': 'Graph\nEncoder',
            'graph_encoder_detail': 'Relation Attention',
            
            # Core components
            'latent_core': 'Universal\nLatent Core',
            'latent_core_detail': 'All-MLP Backbone & HyperMixer',
            'memory': 'Hierarchical\nMemory',
            'memory_detail': 'Short-term, Long-term\nPersistent, Associative',
            'routing': 'Dynamic\nRouting',
            'routing_detail': 'Mixture-of-Experts\nConditional Activation',
            'symbolic': 'Neurosymbolic\nModule',
            'symbolic_detail': 'Rule Engine & Bayesian Network',
            'cross_modal': 'Cross-Modal\nAttention',
            'cross_modal_detail': 'Adaptive Fusion\nMultimodal Interaction',
            'continual': 'Continual\nLearning',
            'continual_detail': 'Forgetting Reduction\nProgressive Adaptation',
            
            # Outputs
            'text_output': 'Text Output',
            'image_output': 'Image Output',
            'audio_output': 'Audio Output',
            'video_output': 'Video Output',
            'graph_output': 'Graph Output',
            
            # Characteristics
            'complexity': 'Complexity: O(n) - Linear with sequence length',
            'params': 'Size: ~335M parameters (60% in encoders)',
            'routing_info': 'Conditional routing: only activates necessary components',
            'memory_info': 'Hierarchical memory for efficient contextual retention',
            'symbolic_info': 'Neurosymbolic components for structured reasoning',
            'backbone_info': 'Hybrid architecture with linear attention and MLP-Mixer',
            
            # Legend
            'legend_encoders': 'Modular Encoders',
            'legend_latent': 'Universal Latent Core',
            'legend_memory': 'Multi-level Memory',
            'legend_routing': 'Adaptive Routing',
            'legend_symbolic': 'Neurosymbolic Module',
            'legend_cross_modal': 'Cross-Modal Attention',
            'legend_continual': 'Continual Learning',
            'legend_output': 'Output Projections',
            'legend_tokenizer': 'Multimodal Tokenizer'
        }
    }
    
    # Utiliser les traductions selon la langue sélectionnée
    if lang not in translations:
        lang = 'fr'  # Langue par défaut
    txt = translations[lang]
    
    # Créer la figure et les axes avec une taille plus grande adaptée aux publications académiques
    fig_width, fig_height = 24, 18  # Taille encore plus grande pour une meilleure lisibilité
    fig_main, ax_main = plt.subplots(figsize=(fig_width, fig_height), facecolor=bg_color)
    
    # Palette de couleurs adaptée aux publications académiques
    colors = {
        'text': '#4c72b0',      # Bleu
        'image': '#55a868',     # Vert
        'audio': '#c44e52',     # Rouge
        'video': '#8172b3',     # Violet
        'graph': '#ccb974',     # Jaune
        'latent': '#dd8452',    # Orange
        'memory': '#64b5cd',    # Bleu ciel
        'output': '#6d904f',    # Vert olive
        'symbolic': '#8c8c8c',  # Gris
        'routing': '#b47cc7',   # Lavande
        'edge': '#505050',      # Gris foncé
    }
    
    # Ajuster les couleurs pour le style moderne (fond sombre)
    if style == "modern":
        for key in colors:
            r, g, b = int(colors[key][1:3], 16)/255, int(colors[key][3:5], 16)/255, int(colors[key][5:7], 16)/255
            r = min(1.0, r*1.3)
            g = min(1.0, g*1.3)
            b = min(1.0, b*1.3)
            colors[key] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    
    # Création de la figure principale
    # Utiliser directement fig_width et fig_height déjà définis
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300, facecolor=bg_color)
    gs = fig.add_gridspec(nrows=18, ncols=24, wspace=0.3, hspace=0.4)
    ax_main = fig.add_subplot(gs[:, :], facecolor=bg_color)
    ax_main.set_xlim(0, fig_width)
    ax_main.set_ylim(0, fig_height)
    ax_main.axis('off')
    
    # Ajouter titre et sous-titre avec taille augmentée
    fig.suptitle(txt['title'], fontsize=36, y=0.98, color=text_color, fontweight='bold')
    ax_main.text(fig_width/2, fig_height-1.5, txt['subtitle'], horizontalalignment='center', 
                verticalalignment='center', fontsize=24, color=text_color, 
                fontweight='normal', alpha=0.95, style='italic')
    
    # Fonction pour créer un bloc avec effets visuels avancés pour les composants
    def create_component_block(x, y, width, height, color, label, detail=None, 
                             alpha=0.8, fontsize=14, detail_fontsize=12, important=False):
        # Définir rect comme variable pour tous les chemins
        rect = None
        
        # Créer ombre portée sous le composant
        if style != "minimal":
            shadow = patches.FancyBboxPatch(
                (x + shadow_offset, y - shadow_offset), width, height,
                boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
                facecolor='black', alpha=0.2,
                edgecolor='none', linewidth=0,
                zorder=1
            )
            ax_main.add_patch(shadow)
        
        # Créer un effet de lueur si activé
        if use_glow_effect and (important or style in ["modern", "dark"]):
            intensity = 1.5 if important else 1.0
            # Calculer une version plus claire de la couleur pour l'effet de lueur
            r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
            glow_color = color
            if style == "dark":
                # Rendre la lueur plus blanche/brillante pour le style sombre
                r, g, b = min(1.0, r*1.7), min(1.0, g*1.7), min(1.0, b*1.7)
                glow_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            
            add_glow_effect(ax_main, x + width/2, y + height/2, 
                         max(width, height) * glow_radius * intensity,
                         glow_color, alpha=glow_alpha*intensity)
        
        # Rectangle principal avec bords arrondis et effets selon le style
        if gradient_style == "radial":
            # Créer un dégradé radial
            r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
            lighter_color = f'#{int(min(r*1.2, 1.0)*255):02x}{int(min(g*1.2, 1.0)*255):02x}{int(min(b*1.2, 1.0)*255):02x}'
            darker_color = f'#{int(r*0.8*255):02x}{int(g*0.8*255):02x}{int(b*0.8*255):02x}'
            
            # Fond avec dégradé radial
            n_circles = 5
            for i in range(n_circles):
                scale = 1 - (i / n_circles)
                # Mélanger les couleurs
                blend_factor = scale
                blend_r = r * blend_factor + float(int(lighter_color[1:3], 16)/255) * (1-blend_factor)
                blend_g = g * blend_factor + float(int(lighter_color[3:5], 16)/255) * (1-blend_factor)
                blend_b = b * blend_factor + float(int(lighter_color[5:7], 16)/255) * (1-blend_factor)
                blend_color = f'#{int(blend_r*255):02x}{int(blend_g*255):02x}{int(blend_b*255):02x}'
                
                inner_rect = patches.FancyBboxPatch(
                    (x + width*(1-scale)/2, y + height*(1-scale)/2), 
                    width*scale, height*scale,
                    boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1*scale),
                    facecolor=blend_color, alpha=alpha,
                    edgecolor='none', linewidth=0,
                    zorder=2
                )
                ax_main.add_patch(inner_rect)
                
            # Définir un rectangle externe pour l'identifier
            rect = patches.FancyBboxPatch(
                (x, y), width, height,
                boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
                facecolor='none', alpha=0,
                edgecolor=color, linewidth=0.5,
                zorder=2
            )
            ax_main.add_patch(rect)
        
        elif gradient_style == "linear":
            # Créer un dégradé linéaire
            r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
            lighter_color = f'#{int(min(r*1.3, 1.0)*255):02x}{int(min(g*1.3, 1.0)*255):02x}{int(min(b*1.3, 1.0)*255):02x}'
            
            # Fond avec dégradé linéaire
            n_layers = 8
            for i in range(n_layers):
                y_offset = i * (height / n_layers)
                layer_height = height / n_layers
                
                # Mélanger les couleurs progressivement
                blend_factor = i / (n_layers-1)
                blend_r = r * (1-blend_factor) + float(int(lighter_color[1:3], 16)/255) * blend_factor
                blend_g = g * (1-blend_factor) + float(int(lighter_color[3:5], 16)/255) * blend_factor
                blend_b = b * (1-blend_factor) + float(int(lighter_color[5:7], 16)/255) * blend_factor
                blend_color = f'#{int(blend_r*255):02x}{int(blend_g*255):02x}{int(blend_b*255):02x}'
                
                layer_rect = patches.Rectangle(
                    (x, y + y_offset), width, layer_height,
                    facecolor=blend_color, alpha=alpha,
                    edgecolor='none', linewidth=0,
                    zorder=2
                )
                ax_main.add_patch(layer_rect)
            
            # Rectangle extérieur pour les bords arrondis
            rect = patches.FancyBboxPatch(
                (x, y), width, height,
                boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
                facecolor='none', alpha=1.0,
                edgecolor=color, linewidth=1.5,
                zorder=3
            )
            ax_main.add_patch(rect)
        
        elif gradient_style == "glow":
            # Effet lueur interne pour style sombre
            rect = patches.FancyBboxPatch(
                (x, y), width, height,
                boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
                facecolor=color, alpha=alpha*0.7,  # Fond légèrement transparent
                edgecolor=color, linewidth=1.5,
                zorder=2
            )
            ax_main.add_patch(rect)
            
            # Ajouter une lueur interne
            r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
            glow_color = f'#{int(min(r*1.5, 1.0)*255):02x}{int(min(g*1.5, 1.0)*255):02x}{int(min(b*1.5, 1.0)*255):02x}'
            
            # Lueur interne comme un rectangle légèrement plus petit
            inner_glow = patches.FancyBboxPatch(
                (x + width*0.1, y + height*0.1), width*0.8, height*0.8,
                boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
                facecolor=glow_color, alpha=alpha*0.3,
                edgecolor='none', linewidth=0,
                zorder=3
            )
            ax_main.add_patch(inner_glow)
        
        else:  # Style minimal ou autre
            # Rectangle simple
            rect = patches.FancyBboxPatch(
                (x, y), width, height,
                boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
                facecolor=color, alpha=alpha,
                edgecolor=color, linewidth=1.5,
                zorder=2
            )
            ax_main.add_patch(rect)
        
        # Ajouter légère ombre au texte (pas en mode sombre)
        if style not in ["modern", "dark"]:
            ax_main.text(x + width/2, y + height/2, label, 
                        horizontalalignment='center', verticalalignment='center', 
                        fontsize=fontsize, color='#333333', alpha=0.3,
                        fontweight='normal', zorder=4)
        
        # Texte principal avec effets selon le style
        text_obj = ax_main.text(x + width/2, y + height/2, label, 
                    horizontalalignment='center', verticalalignment='center', 
                    fontsize=fontsize, color=text_color,
                    fontweight='bold' if important else 'medium', 
                    zorder=5)
        
        # Ajouter un effet de contour au texte pour styles sombres
        if style in ["modern", "dark"]:
            text_obj.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black', alpha=0.7)])
        
        # Ajouter détails si fournis
        if detail:
            detail_obj = ax_main.text(x + width/2, y + height/2 + 0.5, detail, 
                       horizontalalignment='center', verticalalignment='center', 
                       fontsize=detail_fontsize, color=text_color,
                       fontweight='normal', alpha=0.7, zorder=5)
            
            # Effet de contour pour le texte de détail en mode sombre
            if style in ["modern", "dark"]:
                detail_obj.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black', alpha=0.6)])
        
        return rect
    
    # Fonction pour créer une flèche stylisée avec effets visuels entre composants
    def create_arrow(start_x, start_y, end_x, end_y, color, arc=0.0, 
                    width=1.5, alpha=0.7, bidirectional=False, zorder=1, 
                    important=False, dashed=False):
        
        # Paramètres avancés selon le style
        if style == "dark" or style == "modern":
            # Effet de lueur autour des flèches importantes pour style sombre
            if important and add_glow:
                # Créer une version plus épaisse et légèrement transparente de la flèche comme lueur
                glow_arrow = patches.FancyArrowPatch(
                    (start_x, start_y), (end_x, end_y),
                    connectionstyle=f"arc3,rad={arc}",
                    arrowstyle="Simple,head_width=9,head_length=12",
                    linewidth=width*2.5, color=color, alpha=alpha*0.4, zorder=zorder-1
                )
                ax_main.add_patch(glow_arrow)
                
                # Ajouter une deuxième lueur encore plus diffuse
                outer_glow = patches.FancyArrowPatch(
                    (start_x, start_y), (end_x, end_y),
                    connectionstyle=f"arc3,rad={arc}",
                    arrowstyle="Simple,head_width=12,head_length=15",
                    linewidth=width*4, color=color, alpha=alpha*0.2, zorder=zorder-2
                )
                ax_main.add_patch(outer_glow)
        
        # Style de ligne selon les paramètres
        if dashed or (line_style == "dashed" and not important):
            ls = (0, (5, 3))  # ligne pointillée
        elif style == "minimal" and not important:
            ls = (0, (3, 2, 1, 2))  # ligne pointillée fine
        else:
            ls = 'solid'
        
        # Flèche principale
        arrow_style = "Simple,head_width=6,head_length=8"
        if important:
            arrow_style = "Fancy,head_width=8,head_length=10,tail_width=0.5"
            width *= 1.3
            
        arrow = patches.FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            connectionstyle=f"arc3,rad={arc}",
            arrowstyle=arrow_style,
            linewidth=width, 
            color=color, 
            alpha=alpha,
            linestyle=ls,
            zorder=zorder
        )
        ax_main.add_patch(arrow)
        
        # Ajouter effet de chemin pour les flèches importantes dans les styles sombres
        if important and style in ["modern", "dark"]:
            arrow.set_path_effects([path_effects.Stroke(linewidth=width+1, foreground='black', alpha=0.3),
                                   path_effects.Normal()])
        
        if bidirectional:
            # Calculer un décalage perpendiculaire pour la flèche de retour
            # Cela crée une meilleure séparation visuelle entre les flèches
            dx = end_x - start_x
            dy = end_y - start_y
            length = np.sqrt(dx**2 + dy**2)
            # Vecteur perpendiculaire normalisé
            if length > 0:
                offset_x = -dy / length * 0.2
                offset_y = dx / length * 0.2
            else:
                offset_x, offset_y = 0, 0.15
            
            # Flèche dans le sens inverse avec décalage
            arrow_back = patches.FancyArrowPatch(
                (end_x + offset_x, end_y + offset_y), 
                (start_x + offset_x, start_y + offset_y),
                connectionstyle=f"arc3,rad={-arc if arc != 0 else arc}",  # Inverser l'arc pour la flèche de retour
                arrowstyle=arrow_style,
                linewidth=width * 0.9,  # Légèrement plus fine
                color=color, 
                alpha=alpha * 0.9,  # Légèrement plus transparente
                linestyle=ls,
                zorder=zorder-1
            )
            ax_main.add_patch(arrow_back)
            
            # Ajouter effet de chemin pour la flèche de retour si importante
            if important and style in ["modern", "dark"]:
                arrow_back.set_path_effects([path_effects.Stroke(linewidth=width+0.8, foreground='black', alpha=0.25),
                                          path_effects.Normal()])
        
        return arrow
    
    # === ENTRÉES (en bas) ===
    modalities = ['text', 'image', 'audio', 'video', 'graph']
    input_y = 3.0  # Position en bas
    input_width = 2.2  # Blocs plus larges
    input_height = 1.2  # Blocs plus hauts
    input_spacing = 1.2  # Encore plus d'espace entre les entrées
    
    # Centrer les entrées
    total_input_width = input_width * len(modalities) + input_spacing * (len(modalities) - 1)
    input_x_start = (fig_width - total_input_width) / 2
    
    input_boxes = {}
    for i, modality in enumerate(modalities):
        x = input_x_start + i * (input_width + input_spacing)
        input_label = txt[f'{modality}_input']
        input_boxes[modality] = (x, input_y, input_width, input_height)
        create_component_block(x, input_y, input_width, input_height, 
                              colors[modality], input_label, alpha=box_alpha)
    
    # === TOKENIZER MULTIMODAL ===
    tokenizer_y = 5.0  # Position plus élevée pour éviter le chevauchement
    tokenizer_width = 18.0  # Plus large
    tokenizer_height = 1.5  # Plus haut
    tokenizer_x = (fig_width - tokenizer_width) / 2  # Centré par rapport à la figure
    
    # Créer le bloc tokenizer
    tokenizer_color = "#7d3c98"  # Violet foncé
    create_component_block(tokenizer_x, tokenizer_y, tokenizer_width, tokenizer_height,
                          tokenizer_color, txt['tokenizer'], 
                          detail=txt['tokenizer_detail'],
                          alpha=box_alpha, important=True)
    
    # Connexions entrées -> tokenizer
    for i, modality in enumerate(modalities):
        in_x, in_y = input_boxes[modality][0] + input_width/2, input_boxes[modality][1] + input_height
        # Calculer le point de destination sur le tokenizer (réparti)
        token_x = tokenizer_x + tokenizer_width * (i + 1) / (len(modalities) + 1)
        token_y = tokenizer_y
        create_arrow(in_x, in_y, token_x, token_y, colors[modality], alpha=edge_alpha*0.8)
    
    # === ENCODEURS SPÉCIALISÉS ===
    encoder_y = 7.5  # Position encore plus haute
    encoder_width = 2.4  # Plus larges
    encoder_height = 1.8  # Plus hauts
    
    encoder_boxes = {}
    for i, modality in enumerate(modalities):
        x = input_x_start + i * (input_width + input_spacing)
        encoder_label = txt[f'{modality}_encoder']
        encoder_detail = txt[f'{modality}_encoder_detail']
        encoder_boxes[modality] = (x, encoder_y, encoder_width, encoder_height)
        create_component_block(x, encoder_y, encoder_width, encoder_height, 
                              colors[modality], encoder_label, 
                              detail=encoder_detail,
                              alpha=box_alpha)
        
        # Connexion tokenizer -> encodeur
        token_x = tokenizer_x + tokenizer_width * (i + 1) / (len(modalities) + 1)
        token_y = tokenizer_y + tokenizer_height
        enc_x, enc_y = x + encoder_width/2, encoder_y
        create_arrow(token_x, token_y, enc_x, enc_y, colors[modality], alpha=edge_alpha)
    
    # === MODULE D'ATTENTION CROSS-MODALE ===
    cross_modal_y = 10.0  # Position encore plus haute
    cross_modal_width = 16.0  # Plus large
    cross_modal_height = 1.8  # Plus haut
    cross_modal_x = (fig_width - cross_modal_width) / 2  # Centré sur la figure
    cross_modal_color = "#9b59b6"  # Violet moyen
    
    # Créer le bloc d'attention cross-modale
    create_component_block(cross_modal_x, cross_modal_y, cross_modal_width, cross_modal_height,
                         cross_modal_color, txt['cross_modal'], 
                         detail=txt['cross_modal_detail'],
                         alpha=box_alpha, important=True)
    
    # Connexions encodeurs -> attention cross-modale
    for i, modality in enumerate(modalities):
        enc_x = encoder_boxes[modality][0] + encoder_width/2
        enc_y = encoder_boxes[modality][1] + encoder_height
        
        cross_x = cross_modal_x + cross_modal_width * (i + 1) / (len(modalities) + 1)
        cross_y = cross_modal_y
        
        # Flèche avec arc adapté à la position
        arc = 0.1 if i < len(modalities)//2 else -0.1
        create_arrow(enc_x, enc_y, cross_x, cross_y, colors[modality], 
                    arc=arc, alpha=edge_alpha*0.9, width=1.6)
    
    # === NOYAU LATENT UNIVERSEL (au centre) ===
    latent_x, latent_y = fig_width/2 - 3.5, 12.5  # Position encore plus haute
    latent_width, latent_height = 7, 4.0  # Plus grand
    
    # Créer un ovale/cercle pour le noyau latent
    latent_ellipse = patches.Ellipse(
        (latent_x + latent_width/2, latent_y + latent_height/2),
        latent_width, latent_height,
        facecolor=colors['latent'], alpha=box_alpha,
        edgecolor=colors['latent'], linewidth=2
    )
    ax_main.add_patch(latent_ellipse)
    
    # Texte du noyau latent
    ax_main.text(latent_x + latent_width/2, latent_y + latent_height/2, 
                txt['latent_core'], horizontalalignment='center', 
                verticalalignment='center', fontsize=18, color=text_color,
                fontweight='bold', zorder=3)
    
    # Détails du noyau latent sous le titre
    ax_main.text(latent_x + latent_width/2, latent_y + latent_height/2 + 0.5, 
                txt['latent_core_detail'], horizontalalignment='center', 
                verticalalignment='center', fontsize=14, color=text_color,
                fontweight='normal', alpha=0.9, zorder=3)
    
    # Connexion attention cross-modale -> noyau latent
    cross_center_x = cross_modal_x + cross_modal_width/2
    cross_center_y = cross_modal_y + cross_modal_height
    latent_center_x = latent_x + latent_width/2
    latent_center_y = latent_y
    
    create_arrow(cross_center_x, cross_center_y, latent_center_x, latent_center_y, 
                cross_modal_color, arc=0.0, alpha=edge_alpha, width=2.0, important=True)
    
    # === MÉMOIRE HIÉRARCHIQUE (à droite) ===
    memory_x, memory_y = latent_x + latent_width + 2.0, latent_y - 0.25
    memory_width, memory_height = 4.0, 3.5  # Encore plus grande
    
    create_component_block(memory_x, memory_y, memory_width, memory_height, 
                          colors['memory'], txt['memory'], 
                          detail=txt['memory_detail'], alpha=box_alpha, important=True)
    
    # Connexion bidirectionnelle latent <-> mémoire
    mem_conn_x1 = latent_x + latent_width
    mem_conn_y1 = latent_y + latent_height/2
    mem_conn_x2 = memory_x
    mem_conn_y2 = memory_y + memory_height/2
    create_arrow(mem_conn_x1, mem_conn_y1, mem_conn_x2, mem_conn_y2, 
                colors['edge'], arc=0.2, alpha=edge_alpha, bidirectional=True, important=True)
    
    # === APPRENTISSAGE CONTINU (au-dessus de la mémoire) ===
    continual_x = memory_x + 0.5
    continual_y = memory_y + memory_height + 1.0
    continual_width = 3.0
    continual_height = 1.8
    continual_color = "#3498db"  # Bleu clair
    
    create_component_block(continual_x, continual_y, continual_width, continual_height,
                         continual_color, txt['continual'],
                         detail=txt['continual_detail'], alpha=box_alpha)
    
    # Connexion bidirectionnelle mémoire <-> apprentissage continu
    cont_conn_x1 = memory_x + memory_width/2
    cont_conn_y1 = memory_y + memory_height
    cont_conn_x2 = continual_x + continual_width/2
    cont_conn_y2 = continual_y
    create_arrow(cont_conn_x1, cont_conn_y1, cont_conn_x2, cont_conn_y2,
                continual_color, arc=0.0, alpha=edge_alpha, bidirectional=True)
    
    # === ROUTAGE DYNAMIQUE (à gauche) ===
    routing_x, routing_y = latent_x - 6.0, latent_y - 0.25
    routing_width, routing_height = 4.0, 3.5  # Même taille que la mémoire pour l'équilibre
    
    create_component_block(routing_x, routing_y, routing_width, routing_height, 
                         colors['routing'], txt['routing'], 
                         detail=txt['routing_detail'], alpha=box_alpha, important=True)
    
    # Connexion bidirectionnelle latent <-> routage
    route_conn_x1 = latent_x
    route_conn_y1 = latent_y + latent_height/2
    route_conn_x2 = routing_x + routing_width
    route_conn_y2 = routing_y + routing_height/2
    create_arrow(route_conn_x1, route_conn_y1, route_conn_x2, route_conn_y2, 
                colors['edge'], arc=-0.2, alpha=edge_alpha, bidirectional=True, important=True)
    
    # === MODULE NEUROSYMBOLIQUE (en bas) ===
    symbolic_x, symbolic_y = latent_x + latent_width/2 - 3.0, latent_y - 5.5
    symbolic_width, symbolic_height = 6.0, 2.8  # Encore plus grand
    
    create_component_block(symbolic_x, symbolic_y, symbolic_width, symbolic_height, 
                          colors['symbolic'], txt['symbolic'], 
                          detail=txt['symbolic_detail'], 
                          alpha=box_alpha, important=True)
    
    # Connexion bidirectionnelle latent <-> symbolique
    sym_conn_x1 = latent_x + latent_width/2
    sym_conn_y1 = latent_y
    sym_conn_x2 = symbolic_x + symbolic_width/2
    sym_conn_y2 = symbolic_y + symbolic_height
    create_arrow(sym_conn_x1, sym_conn_y1, sym_conn_x2, sym_conn_y2, 
                colors['edge'], arc=0.0, alpha=edge_alpha, bidirectional=True, important=True)
    
    # === SORTIES MULTIMODALES (en haut) ===
    output_y = fig_height - 2.0  # Position tout en haut
    output_width = 2.8  # Plus larges
    output_height = 1.5  # Plus hautes
    output_spacing = 1.5  # Encore plus d'espace entre les sorties
    
    # Centrer les sorties comme les entrées
    output_x_start = input_x_start
    
    # === PROJECTION DE SORTIE MULTIMODALE ===
    projection_y = fig_height - 4.0  # Position en haut
    projection_width = 18.0  # Plus large
    projection_height = 1.8  # Plus haute
    projection_x = (fig_width - projection_width) / 2  # Centré sur la figure
    projection_color = "#27ae60"  # Vert foncé
    
    # Créer le bloc de projection de sortie
    create_component_block(projection_x, projection_y, projection_width, projection_height,
                          projection_color, "Projection Multimodale",
                          alpha=box_alpha, important=True)
    
    # Connexion noyau latent -> projection
    latent_top_x = latent_x + latent_width/2
    latent_top_y = latent_y + latent_height/2 + (latent_height/2)
    proj_center_x = projection_x + projection_width/2
    proj_center_y = projection_y
    
    create_arrow(latent_top_x, latent_top_y, proj_center_x, proj_center_y,
                colors['latent'], arc=0.0, alpha=edge_alpha, width=2.0, important=True)
    
    # Ajouter les sorties et leurs connexions
    output_boxes = {}
    for i, modality in enumerate(modalities):
        x = output_x_start + i * (output_width + output_spacing)
        output_label = txt[f'{modality}_output']
        output_boxes[modality] = (x, output_y, output_width, output_height)
        create_component_block(x, output_y, output_width, output_height, 
                             colors['output'], output_label, alpha=box_alpha)
        
        # Connexion projection -> sortie
        proj_x = projection_x + projection_width * (i + 1) / (len(modalities) + 1)
        proj_y = projection_y + projection_height
        out_x, out_y = x + output_width/2, output_y
        
        create_arrow(proj_x, proj_y, out_x, out_y, colors['output'], 
                    arc=0.0, alpha=edge_alpha, width=1.5)
    
    # === LÉGENDE ET ANNOTATIONS TECHNIQUES ===
    # Zone de légende (repositionnée sur le côté droit dans un encadré distinct)
    legend_x, legend_y = fig_width - 7.0, 7.0  # Position ajustée pour la nouvelle disposition
    legend_items = [
        (txt['legend_tokenizer'], "#7d3c98"),  # Tokenizer
        (txt['legend_encoders'], colors['text']),  # Encodeurs
        (txt['legend_cross_modal'], "#9b59b6"),  # Cross-Modal
        (txt['legend_latent'], colors['latent']),  # Noyau latent
        (txt['legend_memory'], colors['memory']),  # Mémoire
        (txt['legend_continual'], "#3498db"),  # Apprentissage continu
        (txt['legend_routing'], colors['routing']),  # Routage
        (txt['legend_symbolic'], colors['symbolic']),  # Module neurosymbolique
        (txt['legend_output'], "#27ae60")  # Projections de sortie
    ]
    legend_width = 4.0  # Plus large
    legend_height = len(legend_items) * 0.9 + 0.6  # Plus haute
    
    # Encadré pour la légende avec coins arrondis
    legend_box = patches.FancyBboxPatch(
        (legend_x - 0.3, legend_y - 0.3), legend_width, legend_height,
        boxstyle=patches.BoxStyle("Round", pad=0.2, rounding_size=0.1),
        facecolor=bg_color, alpha=0.9 if style in ["modern", "dark"] else 0.7,
        edgecolor=text_color, linewidth=0.7, zorder=10
    )
    ax_main.add_patch(legend_box)
    
    # Ajouter un titre à la légende
    legend_title = "Légende" if lang == "fr" else "Legend"
    ax_main.text(legend_x + legend_width/2 - 0.3, legend_y - 0.1, legend_title,
               horizontalalignment='center', fontsize=12, fontweight='bold',
               color=text_color, zorder=11)
    
    # Ligne de séparation sous le titre
    legend_sep = patches.Rectangle(
        (legend_x - 0.1, legend_y), legend_width - 0.4, 0.02,
        facecolor=text_color, alpha=0.3, zorder=11
    )
    ax_main.add_patch(legend_sep)
    
    # Ajouter les éléments de la légende avec un style amélioré
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y + 0.2 + i * 0.5  # Espacement plus serré mais confortable
        
        # Rectangle coloré avec coins arrondis pour une meilleure esthétique
        rect = patches.FancyBboxPatch(
            (legend_x - 0.1, y_pos), 0.35, 0.35,
            boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
            linewidth=1.0, edgecolor=color, 
            facecolor=color, alpha=0.85,
            zorder=11
        )
        ax_main.add_patch(rect)
        
        # Texte plus lisible avec effet de contour pour les styles sombres
        legend_text = ax_main.text(legend_x + 0.35, y_pos + 0.17, label, 
                    verticalalignment='center', fontsize=11, 
                    color=text_color, fontweight='medium',
                    zorder=11)
                    
        if style in ["modern", "dark"]:
            legend_text.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black', alpha=0.5)])
    
    # Annotations techniques complètement redessinées dans un encadré distinct en bas
    info_width = 22.0  # Encore plus large
    info_height = 3.5  # Encore plus haut
    info_x = (fig_width - info_width) / 2  # Centré horizontalement
    info_y = 0.8                   # Positioné légèrement plus haut
    
    # Créer un encadré distinct pour les annotations techniques
    tech_box = patches.FancyBboxPatch(
        (info_x, info_y), info_width, info_height,
        boxstyle=patches.BoxStyle("Round", pad=0.3, rounding_size=0.2),
        facecolor=bg_color, alpha=0.9 if style in ["modern", "dark"] else 0.7,
        edgecolor=text_color, linewidth=1.0,
        zorder=15
    )
    ax_main.add_patch(tech_box)
    
    # Ajouter un titre à l'encadré
    title_y = info_y + info_height - 0.6
    title_text = "Fondements Techniques NeuroLite" if lang == "fr" else "NeuroLite Technical Foundations"
    ax_main.text(info_x + info_width/2, title_y, title_text, 
                fontsize=18, color=text_color, fontweight='bold',
                horizontalalignment='center', verticalalignment='center',
                zorder=16)
    
    # Ligne de séparation sous le titre
    separator = patches.Rectangle(
        (info_x + 0.5, title_y - 0.25), info_width - 1.0, 0.02,
        facecolor=text_color, alpha=0.3, zorder=16
    )
    ax_main.add_patch(separator)
    
    # Texte des annotations avec espacement et style améliorés - colonne gauche
    text_y = info_y + 1.5
    ax_main.text(info_x + 0.8, text_y, txt['complexity'], 
                fontsize=16, alpha=1.0, 
                color=text_color, fontweight='medium', zorder=16)
    ax_main.text(info_x + 0.8, text_y - 0.6, txt['params'], 
                fontsize=16, alpha=1.0, 
                color=text_color, fontweight='medium', zorder=16)
    ax_main.text(info_x + 0.8, text_y - 1.2, txt['routing_info'], 
                fontsize=16, alpha=1.0, 
                color=text_color, fontweight='medium', zorder=16)
    
    # Colonne centrale
    ax_main.text(info_x + 7.5, text_y, txt['memory_info'], 
                fontsize=16, alpha=1.0, 
                color=text_color, fontweight='medium', zorder=16)
    ax_main.text(info_x + 7.5, text_y - 0.6, txt['symbolic_info'], 
                fontsize=16, alpha=1.0, 
                color=text_color, fontweight='medium', zorder=16)
    ax_main.text(info_x + 7.5, text_y - 1.2, txt['backbone_info'], 
                fontsize=16, alpha=1.0, 
                color=text_color, fontweight='medium', zorder=16)
    
    # Appliquer la texture de fond si demandé
    if add_texture:
        create_texture_background(ax_main, style, density=texture_density, alpha=texture_alpha)
    
    # Ajouter une signature discrète
    signature_text = "Architecture NeuroLite v1.2" 
    plt.figtext(0.98, 0.02, signature_text, fontsize=8, color=text_color, alpha=0.5,
               ha='right', va='bottom')
    
    # Appliquer un filtre de post-traitement selon le style
    if style == "dark" and high_quality:
        # Effet spécial pour le style sombre - lueurs aux intersections
        key_points = [
            (latent_x + latent_width/2, latent_y + latent_height/2),  # Centre du noyau latent
            (memory_x + memory_width/2, memory_y + memory_height/2),     # Centre de la mémoire
            (routing_x + routing_width/2, routing_y + routing_height/2), # Centre du routage
            (symbolic_x + symbolic_width/2, symbolic_y + symbolic_height/2) # Centre symbolique
        ]
        
        for x, y in key_points:
            # Ajouter une lueur discrète aux points clés
            if use_glow_effect:
                add_glow_effect(ax_main, x, y, 0.8, '#ffffff', alpha=0.15, n_circles=10)
    
    # Ajouter un filigrane à peine visible (watermark)
    if high_quality:
        watermark_text = "NeuroLite AI"
        # Utiliser une police de taille appropriée
        font_properties = {'family': 'sans-serif', 'weight': 'normal', 'size': 40}
        watermark = TextPath((0, 0), watermark_text, size=0.5, prop=font_properties)
        transform = transforms.Affine2D().rotate_deg(45).translate(8, 6)
        watermark = transform.transform_path(watermark)
        patch = patches.PathPatch(watermark, facecolor=text_color, alpha=0.03, 
                               edgecolor='none', zorder=0)
        ax_main.add_patch(patch)
    
    # Optimiser les paramètres d'export selon la qualité demandée
    if high_quality:
        # Export haute qualité pour publications
        plt.savefig(save_path, dpi=dpi_export, bbox_inches='tight', facecolor=bg_color,
                  edgecolor='none', pad_inches=0.2, metadata={'Creator': 'NeuroLite Framework'})
    else:
        # Export standard
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    
    print(f"Diagramme d'architecture NeuroLite avancé sauvegardé à {save_path}")
    return os.path.abspath(save_path)


if __name__ == "__main__":
    import argparse
    
    # Configurer l'analyseur d'arguments
    parser = argparse.ArgumentParser(description='Générateur de diagramme NeuroLite Architecture')
    parser.add_argument('--style', choices=["academic", "modern", "minimal", "dark"], default="modern",
                        help='Style visuel du diagramme (academic, modern, minimal, dark)')
    parser.add_argument('--lang', choices=["fr", "en"], default="fr",
                        help='Langue des annotations (fr, en)')
    parser.add_argument('--quality', choices=["high", "normal"], default="high",
                        help='Qualité de l’export (high, normal)')
    parser.add_argument('--texture', action='store_true', default=True,
                        help='Ajouter une texture de fond')
    parser.add_argument('--glow', action='store_true', default=True,
                        help='Ajouter des effets de lueur')
    parser.add_argument('--all', action='store_true',
                        help='Générer toutes les combinaisons possibles de styles et langues')
    parser.add_argument('--dpi', type=int, default=450,
                        help='Résolution d’export en DPI (300-600 recommandé)')
    
    # Analyser les arguments
    args = parser.parse_args()
    
    # Créer des versions dans différents styles et langues
    styles = ["academic", "modern", "minimal", "dark"]
    languages = ["fr", "en"]
    
    # Options de génération
    high_quality_export = args.quality == "high"
    generate_all = args.all
    add_texture_effect = args.texture
    use_glow_effects = args.glow
    custom_dpi = args.dpi
    
    # Version par défaut à générer et afficher
    default_style = args.style    # Style par défaut: academic, modern, minimal, dark
    default_lang = args.lang      # Langue par défaut: fr, en
    
    # Créer les dossiers de sortie si nécessaires
    output_dir = Path("architectures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Générer le diagramme par défaut
    default_filename = f"architectures/neurolite_architecture_{default_style}_{default_lang}.png"
    print(f"\nGénération du diagramme principal en style '{default_style}' et langue '{default_lang}'...")
    diagram_path = create_neurolite_architecture_diagram(
        save_path=default_filename,
        style=default_style,
        lang=default_lang,
        add_texture=add_texture_effect,
        use_glow_effect=use_glow_effects,
        high_quality=high_quality_export
    )
    
    # Générer toutes les combinaisons si demandé
    if generate_all:
        print(f"\nGénération de toutes les versions dans différents styles et langues...")
        for style in styles:
            for lang in languages:
                # Sauter la version par défaut déjà générée
                if style == default_style and lang == default_lang:
                    continue
                    
                filename = f"architectures/neurolite_architecture_{style}_{lang}.png"
                print(f"  - Génération de la version {style} en {lang}...")
                create_neurolite_architecture_diagram(
                    save_path=filename,
                    style=style,
                    lang=lang,
                    add_texture=add_texture_effect,
                    use_glow_effect=use_glow_effects,
                    high_quality=high_quality_export
                )
    
    # Ouvrir la version principale générée
    try:
        print(f"\nOuverture du diagramme généré: {diagram_path}")
        if platform.system() == 'Windows':
            os.startfile(diagram_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', diagram_path])
        else:  # Linux
            subprocess.run(['xdg-open', diagram_path])
        print(f"Image ouverte avec le visualiseur par défaut")
    except Exception as e:
        print(f"L'image a été créée mais n'a pas pu être ouverte automatiquement: {e}")
    
    print("\nPour générer d'autres styles, modifiez les variables au début du bloc main.\n")
    print("Styles disponibles: academic, modern, minimal, dark")
    print("Langues disponibles: fr, en")
    print("Options: high_quality_export, add_texture_effect, add_glow_effect, generate_all")
