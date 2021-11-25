<template lang="pug">
table.board
  tr
    td {{ board[0][0]  }}
    td {{ board[0][1]  }}
    td {{ board[0][2]  }}
  tr
    td {{ board[1][0]  }}
    td {{ board[1][1]  }}
    td {{ board[1][2]  }}
  tr
    td {{ board[2][0]  }}
    td {{ board[2][1]  }}
    td {{ board[2][2]  }}
</template>

<script>
export default {
  name: "Board",
  data() {
    return {
      board: [
        ["", "X", "X"],
        ["", "O", "O"],
        ["", "", ""],
      ],
    };
  },
};
</script>

<style>
.board {
  background: #44ebeb;
  border: 1px solid black;
  border-collapse: collapse;
  width: 75px;
  height: 75px;
}
.board tr td {
    border: 1px solid black;
    width: 25px;
    height: 25px;
}
</style>