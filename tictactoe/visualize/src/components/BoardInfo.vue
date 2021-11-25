<template lang="pug">
div
  div
    Board
  .info
    p turn: {{ player }}
    p network evaluation: {{ network_evaluation }}
    p network policy:
    p {{ network_policy }}
    p mcts policy:
    p {{ mcts_policy }}
    p visits: {{ mcts_visits }}
</template>

<script>
import Board from "./Board.vue";
export default {
  name: "BoardInfo",
  components: { Board },
  data() {
    return {
      player: "cross",
      network_evaluation: 0,
      network_policy: [0.0, 0.1, 0.23, 0.0, 0.1, 0.23, 0.0, 0.1, 0.23],
      mcts_policy: [0.0, 0.1, 0.23, 0.0, 0.1, 0.23, 0.0, 0.1, 0.23],
      mcts_visits: 10,
      show_children: true,
      children: [
        {
          player: "circle",
          network_evaluation: 0,
          network_policy: [0.0, 0.1, 0.23, 0.0, 0.1, 0.23, 0.0, 0.1, 0.23],
          mcts_visits: 3,
          show_children: false,
          children: [],
        },
      ],
    };
  },
};
</script>

<style>
.info p {
  margin: 1px;
}
</style>